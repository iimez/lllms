import path from 'node:path'
import fs from 'node:fs'
import {
	EngineContext,
	FileDownloadProgress,
	ModelConfig,
	EngineImageToTextArgs,
	EngineSpeechToTextArgs,
	EngineTextCompletionResult,
	EngineTextCompletionArgs,
	EngineEmbeddingArgs,
	EngineEmbeddingResult,
	ImageEmbeddingInput,
	TransformersJsModel,
	TextEmbeddingInput,
} from '#lllms/types/index.js'
import {
	env,
	AutoModel,
	AutoProcessor,
	AutoTokenizer,
	RawImage,
	TextStreamer,
	mean_pooling,
} from '@huggingface/transformers'
import { LogLevels } from '#lllms/lib/logger.js'
import { acquireFileLock } from '#lllms/lib/acquireFileLock.js'
import { decodeAudio } from '#lllms/lib/audio.js'

// TODO transformers.js types currently hard to fix, until v3 is released on npm (with typedefs)

interface TransformersJsModelComponents {
	model?: any
	processor?: any
	tokenizer?: any
}

interface TransformersJsInstance {
	textModel?: TransformersJsModelComponents
	visionModel?: TransformersJsModelComponents
	speechModel?: TransformersJsModelComponents
}

interface ModelFile {
	file: string
	size: number
}

// interface TransformersJsModelMeta {
// 	modelType: string
// 	files: ModelFile[]
// }

export interface TransformersJsModelConfig extends ModelConfig {
	location: string
	url: string
	textModel?: TransformersJsModel
	visionModel?: TransformersJsModel
	speechModel?: TransformersJsModel
	device?: {
		gpu?: boolean | 'auto' | (string & {})
	}
}

function checkModelExists(config: TransformersJsModelConfig) {
	if (!fs.existsSync(config.location)) {
		return false
	}
	if (!fs.existsSync(config.location + '/onnx')) {
		return false
	}
	
	const checkDTypeExists = (dtype: string | Record<string, string>) => {
		// TODO needs more work, does not handle all cases
		if (typeof dtype === 'string') {
			if (dtype === 'fp32') {
				const expectedFile = `${config.location}/onnx/encoder_model.onnx`
				if (!fs.existsSync(expectedFile)) {
					console.debug('missing', dtype, expectedFile)
					return false
				}
			}
		} else if (typeof dtype === 'object') {
			for (const fileName in dtype) {
				const dataType = dtype[fileName]
				let expectedFile = `${config.location}/onnx/${fileName}_${dataType}.onnx`
				if (dataType === 'fp32') {
					expectedFile = `${config.location}/onnx/${fileName}.onnx`
				}
				if (!fs.existsSync(expectedFile)) {
					console.debug('missing', dtype, expectedFile)
					return false
				}
			}
		}
		return true
	}
	if (config.textModel?.dtype) {
		const notExisting = checkDTypeExists(config.textModel.dtype)
		if (!notExisting) {
			return false
		}
	}
	if (config.visionModel?.dtype) {
		const notExisting = checkDTypeExists(config.visionModel.dtype)
		if (!notExisting) {
			return false
		}
	}
	if (config.speechModel?.dtype) {
		const notExisting = checkDTypeExists(config.speechModel.dtype)
		if (!notExisting) {
			return false
		}
	}
	return true
}

async function remoteFileExists(url: string): Promise<boolean> {
	try {
		const response = await fetch(url, { method: 'HEAD' })
		return response.ok
	} catch (error) {
		console.error('Error checking remote file:', error)
		return false
	}
}

export const autoGpu = true

export async function prepareModel(
	{ config, log }: EngineContext<TransformersJsModelConfig>,
	onProgress?: (progress: FileDownloadProgress) => void,
	signal?: AbortSignal,
) {
	fs.mkdirSync(config.location, { recursive: true })
	const clearFileLock = await acquireFileLock(config.location + '/', signal)
	if (signal?.aborted) {
		return
	}
	log(LogLevels.info, `Preparing transformers.js model at ${config.location}`, {
		model: config.id,
	})
	if (!config.url) {
		throw new Error(`Missing URL for model ${config.id}`)
	}

	const parsedUrl = new URL(config.url)
	// TODO support other hostnames than huggingface.co?
	const urlSegments = parsedUrl.pathname.split('/')
	const org = urlSegments[1]
	const repo = urlSegments[2]
	const branch = urlSegments[4] || 'main'

	const modelId = `${org}/${repo}`
	const modelFiles: ModelFile[] = []
	const configMeta: any = {}

	const downloadModelFiles = async (modelOpts: TransformersJsModel) => {
		const modelClass = modelOpts.modelClass ?? AutoModel
		const downloadPromises = []
		const modelDownloadPromise = modelClass.from_pretrained(modelId, {
			revision: branch,
			dtype: modelOpts.dtype,
			// use_external_data_format: true, // https://github.com/xenova/transformers.js/blob/38a3bf6dab2265d9f0c2f613064535863194e6b9/src/models.js#L205-L207
			progress_callback: (progress: any) => {
				if (onProgress && progress.status === 'progress') {
					onProgress({
						file: env.cacheDir + progress.file,
						loadedBytes: progress.loaded,
						totalBytes: progress.total,
					})
				}
			},
		})
		downloadPromises.push(modelDownloadPromise)

		const hasTokenizer = await remoteFileExists(`${config.url}/blob/${branch}/tokenizer.json`)
		if (hasTokenizer) {
			const TokenizerClass = modelOpts.tokenizerClass ?? AutoTokenizer
			const tokenizerDownload = TokenizerClass.from_pretrained(modelId, {
				revision: branch,
				// use_external_data_format: true,
			})
			downloadPromises.push(tokenizerDownload)
		}

		if (modelOpts.processor) {
			const ProcessorClass = modelOpts.processorClass ?? AutoProcessor
			if (typeof modelOpts.processor === 'string') {
				const processorDownload = ProcessorClass.from_pretrained(modelId)
				downloadPromises.push(processorDownload)
			} else {
				const processorDownload = ProcessorClass.from_pretrained(modelId, {
					revision: branch,
					// use_external_data_format: true,
				})
				downloadPromises.push(processorDownload)
			}
		}
		return await Promise.all(downloadPromises)
	}

	if (!checkModelExists(config)) {
		if (!config.url) {
			throw new Error(`Cannot download "${config.id}" - no URL configured`)
		}
		log(LogLevels.info, `Downloading ${config.id}`, {
			url: config.url,
			location: config.location,
			modelId: modelId,
			branch,
		})
		const modelDownloadPromises = []
		if (config.textModel) {
			modelDownloadPromises.push(downloadModelFiles(config.textModel))
		}
		if (config.visionModel) {
			modelDownloadPromises.push(downloadModelFiles(config.visionModel))
		}
		if (config.speechModel) {
			modelDownloadPromises.push(downloadModelFiles(config.speechModel))
		}

		const models = await Promise.all(modelDownloadPromises)
		for (const model of models) {
			for (const modelComponent of model) {
				if (modelComponent.dispose) {
					modelComponent.dispose()
				}
			}
		}
		if (signal?.aborted) {
			return
		}
		const files = fs.readdirSync(env.cacheDir, { recursive: true })
		for (const file of files) {
			const filePath = file.toString()
			const sourceFile = path.join(env.cacheDir, filePath)
			const sourceStat = fs.statSync(sourceFile)
			let targetFile = path.join(config.location, path.basename(sourceFile))
			const isONNXFile = filePath.match(/\/onnx\/.+\.onnx$/)?.length
			if (isONNXFile) {
				const targetDir = path.join(config.location, '/onnx/')
				fs.mkdirSync(targetDir, { recursive: true })
				targetFile = path.join(targetDir, path.basename(sourceFile))
			}

			const targetExists = fs.existsSync(targetFile)
			modelFiles.push({
				file: targetFile,
				size: sourceStat.size,
			})
			if (targetExists) {
				const targetStat = fs.statSync(targetFile)
				if (sourceStat.size === targetStat.size) {
					continue
				}
			}
			if (sourceStat.isDirectory()) {
				continue
			}
			fs.copyFileSync(sourceFile, targetFile)
			if (targetFile.endsWith('.json')) {
				const key = path.basename(targetFile).replace('.json', '')
				configMeta[key] = JSON.parse(fs.readFileSync(targetFile, 'utf8'))
			}
		}
	} else {
		const files = fs.readdirSync(config.location, { recursive: true })
		for (const file of files) {
			const targetFile = path.join(config.location, file.toString())
			const targetStat = fs.statSync(targetFile)
			modelFiles.push({
				file: targetFile,
				size: targetStat.size,
			})
			if (targetFile.endsWith('.json')) {
				const key = path.basename(targetFile).replace('.json', '')
				configMeta[key] = JSON.parse(fs.readFileSync(targetFile, 'utf8'))
			}
		}
	}
	clearFileLock()
	return {
		files: modelFiles,
		...configMeta,
	}
}

export async function createInstance(
	{ config, log }: EngineContext<TransformersJsModelConfig>,
	signal?: AbortSignal,
) {
	let modelPath = config.location
	if (!modelPath.endsWith('/')) {
		modelPath += '/'
	}

	const loadModel = async (modelOpts: TransformersJsModel) => {
		const modelClass = modelOpts.modelClass ?? AutoModel
		const loadPromises = []
		const modelPromise = modelClass.from_pretrained(modelPath, {
			local_files_only: true,
			cache_dir: '/',
			dtype: modelOpts.dtype,
			device: config.device?.gpu ? 'gpu' : 'cpu',
		})
		loadPromises.push(modelPromise)

		const TokenizerClass = modelOpts.tokenizerClass ?? AutoTokenizer
		const tokenizerPromise = TokenizerClass.from_pretrained(modelPath, {
			local_files_only: true,
			cache_dir: '/',
		})
		loadPromises.push(tokenizerPromise)

		const ProcessorClass = modelOpts.processorClass ?? AutoProcessor
		if (typeof modelOpts.processor === 'string') {
			const processorDownload = ProcessorClass.from_pretrained(
				modelOpts.processor,
			)
			loadPromises.push(processorDownload)
		} else {
			const processorDownload = ProcessorClass.from_pretrained(modelPath, {
				local_files_only: true,
				cache_dir: '/',
			})
			loadPromises.push(processorDownload)
		}
		const results = await Promise.all(loadPromises)
		return {
			model: results[0],
			tokenizer: results[1],
			processor: results[2],
		}
	}
	const modelLoadPromises = []
	const noModelConfigured =
		!config.textModel && !config.visionModel && !config.speechModel

	if (config.textModel || noModelConfigured) {
		modelLoadPromises.push(loadModel(config.textModel || {}))
	} else {
		modelLoadPromises.push(Promise.resolve(undefined))
	}
	if (config.visionModel) {
		modelLoadPromises.push(loadModel(config.visionModel))
	} else {
		modelLoadPromises.push(Promise.resolve(undefined))
	}
	if (config.speechModel) {
		modelLoadPromises.push(loadModel(config.speechModel))
	} else {
		modelLoadPromises.push(Promise.resolve(undefined))
	}

	const models = await Promise.all(modelLoadPromises)
	const instance: TransformersJsInstance = {
		textModel: models[0],
		visionModel: models[1],
		speechModel: models[2],
	}

	// TODO preload whisper / any speech to text?
	// await model.generate({
	// 	input_features: full([1, 80, 3000], 0.0),
	// 	max_new_tokens: 1,
	// });

	return instance
}

export async function disposeInstance(instance: TransformersJsInstance) {
	if (instance.textModel) {
		instance.textModel.model.dispose()
	}
	if (instance.visionModel) {
		instance.visionModel.model.dispose()
	}
	if (instance.speechModel) {
		instance.speechModel.model.dispose()
	}
}

export async function processTextCompletionTask(
	{
		request,
		config,
		log,
		onChunk,
	}: EngineTextCompletionArgs<TransformersJsModelConfig>,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
): Promise<EngineTextCompletionResult> {
	if (!request.prompt) {
		throw new Error('Prompt is required for text completion.')
	}
	const inputTokens = instance.textModel!.tokenizer(request.prompt)
	const outputTokens = await instance.textModel!.model.generate({
		...inputTokens,
		max_new_tokens: request.maxTokens ?? 128,
	})
	const outputText = instance.textModel!.tokenizer.batch_decode(outputTokens, {
		skip_special_tokens: true,
	})

	return {
		finishReason: 'eogToken',
		text: outputText,
		promptTokens: inputTokens.length,
		completionTokens: outputTokens.length,
		totalTokens: inputTokens.length + outputTokens.length,
	}
}

// see https://github.com/xenova/transformers.js/blob/v3/src/utils/tensor.js
// https://github.com/xenova/transformers.js/blob/v3/src/pipelines.js#L1284
export async function processEmbeddingTask(
	{ request, config }: EngineEmbeddingArgs<TransformersJsModelConfig>,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
): Promise<EngineEmbeddingResult> {
	if (!request.input) {
		throw new Error('Input is required for embedding.')
	}
	const inputs = Array.isArray(request.input) ? request.input : [request.input]
	const normalizedInputs: Array<TextEmbeddingInput | ImageEmbeddingInput> =
		inputs.map((input) => {
			if (typeof input === 'string') {
				return {
					type: 'text',
					content: input,
				}
			} else if (input.type) {
				return input
			} else {
				throw new Error('Invalid input type')
			}
		})

	const embeddings: Float32Array[] = []
	let inputTokens = 0

	const applyPooling = (result: any, pooling: string, modelInputs: any) => {
		if (pooling === 'mean') {
			return mean_pooling(result, modelInputs.attention_mask)
		} else if (pooling === 'cls') {
			return result.slice(null, 0)
		} else {
			throw Error(`Pooling method '${pooling}' not supported.`)
		}
	}

	const truncateDimensions = (result: any, dimensions: number) => {
		const truncatedData = new Float32Array(dimensions)
		truncatedData.set(result.data.slice(0, dimensions))
		return truncatedData
	}

	for (const embeddingInput of normalizedInputs) {
		if (signal?.aborted) {
			break
		}
		let result
		let modelInputs
		if (embeddingInput.type === 'text') {
			modelInputs = instance.textModel!.tokenizer(embeddingInput.content, {
				padding: true, // pads input if it is shorter than context window
				truncation: true, // truncates input if it exceeds context window
			})
			inputTokens += modelInputs.input_ids.size
			const modelOutputs = await instance.textModel!.model(modelInputs)
			result =
				modelOutputs.last_hidden_state ??
				modelOutputs.logits ??
				modelOutputs.token_embeddings ??
				modelOutputs.text_embeds
		} else if (embeddingInput.type === 'image') {
			let image: RawImage
			if (embeddingInput.url) {
				image = await RawImage.fromURL(embeddingInput.url)
			} else if (embeddingInput.file) {
				const fileBuffer = fs.readFileSync(embeddingInput.file)
				const blob = new Blob([fileBuffer])
				image = await RawImage.fromBlob(blob)
			} else {
				throw new Error('Invalid image input')
			}
			modelInputs = await instance.visionModel!.processor(image)
			const modelOutputs = await instance.visionModel!.model(modelInputs)
			result =
				modelOutputs.last_hidden_state ??
				modelOutputs.logits ??
				modelOutputs.image_embeds
		}

		if (request.pooling) {
			result = applyPooling(result, request.pooling, modelInputs)
		}
		if (request.dimensions && result.data.length > request.dimensions) {
			embeddings.push(truncateDimensions(result, request.dimensions))
		} else {
			embeddings.push(result.data)
		}
	}

	return {
		embeddings,
		inputTokens,
	}
}

export async function processImageToTextTask(
	{ request, config, log }: EngineImageToTextArgs,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
) {
	let image: any = request.image

	if (request.url) {
		image = await RawImage.fromURL(request.url)
	} else if (request.file) {
		const fileBuffer = fs.readFileSync(request.file)
		const blob = new Blob([fileBuffer])
		image = await RawImage.fromBlob(blob)
	}
	
	if (!image) {
		throw new Error('No image provided')
	}
	
	if (signal?.aborted) {
		return
	}

	// Process inputs
	let textInputs = {}
	if (request.prompt) {
		textInputs = instance.visionModel!.tokenizer(request.prompt)
	}
	const visionInputs = await instance.visionModel!.processor(image)
	const outputTokens = await instance.visionModel!.model.generate({
		...textInputs,
		...visionInputs,
		max_new_tokens: request.maxTokens ?? 128,
	})
	const outputText = instance.visionModel!.tokenizer.batch_decode(outputTokens, {
		skip_special_tokens: true,
	})

	return {
		text: outputText[0],
	}
}

async function readAudioFile(filePath: string) {
	const WHISPER_SAMPLING_RATE = 16_000
	const MAX_AUDIO_LENGTH = 30 // seconds
	const MAX_SAMPLES = WHISPER_SAMPLING_RATE * MAX_AUDIO_LENGTH
	// Read the file into a buffer
	const fileBuffer = fs.readFileSync(filePath)

	// Decode the audio data
	let decodedAudio = await decodeAudio(fileBuffer, WHISPER_SAMPLING_RATE)

	// Trim the audio data if it exceeds MAX_SAMPLES
	if (decodedAudio.length > MAX_SAMPLES) {
		decodedAudio = decodedAudio.slice(-MAX_SAMPLES)
	}

	return decodedAudio
}

// see examples
// https://huggingface.co/docs/transformers.js/guides/node-audio-processing
// https://github.com/xenova/transformers.js/tree/v3/examples/node-audio-processing
export async function processSpeechToTextTask(
	{ request, onChunk }: EngineSpeechToTextArgs,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
) {
	const streamer = new TextStreamer(instance.speechModel!.tokenizer, {
		skip_prompt: true,
		// skip_special_tokens: true,
		callback_function: (output: any) => {
			if (onChunk) {
				onChunk({ text: output })
			}
		},
	})
	let inputs
	if (request.file) {
		const audio = await readAudioFile(request.file)
		inputs = await instance.speechModel!.processor(audio)
	}

	const outputs = await instance.speechModel!.model.generate({
		...inputs,
		max_new_tokens: request.maxTokens ?? 128,
		language: request.language ?? 'en',
		streamer,
	})

	const outputText = instance.speechModel!.tokenizer.batch_decode(outputs, {
		skip_special_tokens: true,
	})

	return {
		text: outputText[0],
	}
}
