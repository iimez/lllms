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
} from '#package/types/index.js'
import {
	env,
	AutoModel,
	AutoProcessor,
	AutoTokenizer,
	RawImage,
	TextStreamer,
	mean_pooling,
	Processor,
	PreTrainedModel,
	PreTrainedTokenizer,
} from '@huggingface/transformers'
import { LogLevels } from '#package/lib/logger.js'
import { acquireFileLock } from '#package/lib/acquireFileLock.js'
import { copyDirectory } from '#package/lib/copyDirectory.js'
import { decodeAudio } from '#package/lib/audio.js'
import { parseModelUrl, remoteFileExists } from './util.js'

interface TransformersJsModelComponents {
	model?: PreTrainedModel
	processor?: Processor
	tokenizer?: PreTrainedTokenizer
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

export const autoGpu = true

let didConfigureEnvironment = false
function configureEnvironment(modelsPath: string) {
	// console.debug({
	// 	cacheDir: env.cacheDir,
	// 	localModelPaths: env.localModelPath,
	// })
	// env.useFSCache = false
	// env.useCustomCache = true
	// env.customCache = new TransformersFileCache(modelsPath)
	env.localModelPath = ''
	didConfigureEnvironment = true
}

async function loadModelComponents(
	modelOpts: TransformersJsModel,
	config: TransformersJsModelConfig,
	modelPath: string,
): Promise<TransformersJsModelComponents> {
	const device = config.device?.gpu ? 'gpu' : 'cpu'
	const modelClass = modelOpts.modelClass ?? AutoModel
	const loadPromises = []
	const modelPromise = modelClass.from_pretrained(modelPath, {
		local_files_only: true,
		device: device,
		dtype: modelOpts.dtype || 'fp32',
	})
	loadPromises.push(modelPromise)
	const tokenizerClass = modelOpts.tokenizerClass ?? AutoTokenizer
	const tokenizerPromise = tokenizerClass.from_pretrained(modelPath, {
		local_files_only: true,
	})
	loadPromises.push(tokenizerPromise)

	const hasPreprocessor = fs.existsSync(modelPath + 'preprocessor_config.json')
	const hasProcessor = fs.existsSync(modelPath + 'processor_config.json')

	if (hasProcessor || hasPreprocessor || modelOpts.processor) {
		const processorClass = modelOpts.processorClass ?? AutoProcessor
		if (modelOpts.processor) {
			const { modelId } = parseModelUrl(modelOpts.processor.url)
			const processorPromise = processorClass.from_pretrained(modelId)
			loadPromises.push(processorPromise)
		} else {
			const processorPromise = processorClass.from_pretrained(modelPath, {
				local_files_only: true,
			})
			loadPromises.push(processorPromise)
		}
	}

	const loadedComponents = await Promise.all(loadPromises)
	const modelComponents: TransformersJsModelComponents = {}
	if (loadedComponents[0]) {
		modelComponents.model = loadedComponents[0] as PreTrainedModel
	}
	if (loadedComponents[1]) {
		modelComponents.tokenizer = loadedComponents[1] as PreTrainedTokenizer
	}
	if (loadedComponents[2]) {
		modelComponents.processor = loadedComponents[2] as Processor
	}
	return modelComponents
}

async function disposeModelComponents(
	modelComponents: TransformersJsModelComponents,
) {
	if (modelComponents.model && 'dispose' in modelComponents.model) {
		await modelComponents.model.dispose()
	}
}

async function validateModel(
	modelOpts: TransformersJsModel,
	config: TransformersJsModelConfig,
	modelPath: string,
): Promise<string | undefined> {
	const modelClass = modelOpts.modelClass ?? AutoModel
	const device = config.device?.gpu ? 'gpu' : 'cpu'
	try {
		const model = await modelClass.from_pretrained(modelPath, {
			local_files_only: true,
			device: device,
			dtype: modelOpts.dtype || 'fp32',
		})
		await model.dispose()
	} catch (error) {
		return `failed to load model: ${error}`
	}
	return undefined
}

async function validateTokenizer(
	modelOpts: TransformersJsModel,
	modelPath: string,
): Promise<string | undefined> {
	const tokenizerClass = modelOpts.tokenizerClass ?? AutoTokenizer
	try {
		await tokenizerClass.from_pretrained(modelPath, {
			local_files_only: true,
		})
	} catch (error) {
		return `failed to load tokenizer: ${error}`
	}
	return undefined
}

async function validateProcessor(
	modelOpts: TransformersJsModel,
	modelPath: string,
): Promise<string | undefined> {
	const processorClass = modelOpts.processorClass ?? AutoProcessor
	try {
		if (modelOpts.processor) {
			const { modelId } = parseModelUrl(modelOpts.processor.url)
			await processorClass.from_pretrained(modelId, {
				local_files_only: true,
			})
		} else {
			await processorClass.from_pretrained(modelPath, {
				local_files_only: true,
			})
		}
	} catch (error) {
		return `failed to load processor: ${error}`
	}
	return undefined
}

interface ComponentValidationErrors {
	model?: string
	tokenizer?: string
	processor?: string
}

interface ModelValidationErrors {
	textModel?: ComponentValidationErrors
	visionModel?: ComponentValidationErrors
	speechModel?: ComponentValidationErrors
}

interface ModelFileValidationResult {
	message: string
	errors?: ModelValidationErrors
}

async function validateModelFiles(
	config: TransformersJsModelConfig,
): Promise<ModelFileValidationResult | undefined> {
	if (!fs.existsSync(config.location)) {
		return {
			message: `model directory does not exist: ${config.location}`,
		}
	}

	let modelPath = config.location
	if (!modelPath.endsWith('/')) {
		modelPath += '/'
	}

	const validateModelComponents = async (modelOpts: TransformersJsModel) => {
		const componentValidationPromises = [
			validateModel(modelOpts, config, modelPath),
			validateTokenizer(modelOpts, modelPath),
		]
		if (modelOpts.processor) {
			componentValidationPromises.push(validateProcessor(modelOpts, modelPath))
		}
		const [model, tokenizer, processor] = await Promise.all(
			componentValidationPromises,
		)
		const result: ComponentValidationErrors = {}
		if (model) result.model = model
		if (tokenizer) result.tokenizer = tokenizer
		if (processor) result.processor = processor
		return result
	}

	const modelValidationPromises: any = {}
	const noModelConfigured =
		!config.textModel && !config.visionModel && !config.speechModel
	if (config.textModel || noModelConfigured) {
		modelValidationPromises.textModel = validateModelComponents(
			config.textModel || {},
		)
	}
	if (config.visionModel) {
		modelValidationPromises.visionModel = validateModelComponents(
			config.visionModel,
		)
	}
	if (config.speechModel) {
		modelValidationPromises.speechModel = validateModelComponents(
			config.speechModel,
		)
	}

	await Promise.all(Object.values(modelValidationPromises))
	const validationErrors: ModelValidationErrors = {}
	const textModelErrors = await modelValidationPromises.textModel
	if (textModelErrors && Object.keys(textModelErrors).length) {
		validationErrors.textModel = textModelErrors
	}
	const visionModelErrors = await modelValidationPromises.visionModel
	if (visionModelErrors && Object.keys(visionModelErrors).length) {
		validationErrors.visionModel = visionModelErrors
	}
	const speechModelErrors = await modelValidationPromises.speechModel
	if (speechModelErrors && Object.keys(speechModelErrors).length) {
		validationErrors.speechModel = speechModelErrors
	}

	if (Object.keys(validationErrors).length > 0) {
		return {
			message: 'failed to load model components',
			errors: validationErrors,
		}
	}
	return undefined
}

export async function prepareModel(
	{ config, log, modelsPath }: EngineContext<TransformersJsModelConfig>,
	onProgress?: (progress: FileDownloadProgress) => void,
	signal?: AbortSignal,
) {
	if (!didConfigureEnvironment && modelsPath) {
		configureEnvironment(modelsPath)
	}
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

	const { modelId, branch } = parseModelUrl(config.url)
	const modelFiles: ModelFile[] = []
	const configMeta: any = {}
	const downloadedFileSet = new Set<string>()

	const downloadModelFiles = async (
		modelOpts: TransformersJsModel,
		requiredComponents: string[] = ['model', 'tokenizer', 'processor'],
	) => {
		const modelClass = modelOpts.modelClass ?? AutoModel
		const downloadPromises: Record<string, Promise<any> | undefined> = {}
		const progressCallback = (progress: any) => {
			if (onProgress && progress.status === 'progress') {
				downloadedFileSet.add(progress.file)
				onProgress({
					file: env.cacheDir + modelId + '/' + progress.file,
					loadedBytes: progress.loaded,
					totalBytes: progress.total,
				})
			}
		}
		if (requiredComponents.includes('model')) {
			const modelDownloadPromise = modelClass.from_pretrained(modelId, {
				revision: branch,
				dtype: modelOpts.dtype || 'fp32',
				progress_callback: progressCallback,
				// use_external_data_format: true, // https://github.com/xenova/transformers.js/blob/38a3bf6dab2265d9f0c2f613064535863194e6b9/src/models.js#L205-L207
			})
			downloadPromises.model = modelDownloadPromise
		}
		if (requiredComponents.includes('tokenizer')) {
			const hasTokenizer = await remoteFileExists(
				`${config.url}/blob/${branch}/tokenizer.json`,
			)
			if (hasTokenizer) {
				const tokenizerClass = modelOpts.tokenizerClass ?? AutoTokenizer
				const tokenizerDownload = tokenizerClass.from_pretrained(modelId, {
					revision: branch,
					progress_callback: progressCallback,
					// use_external_data_format: true,
				})
				downloadPromises.tokenizer = tokenizerDownload
			}
		}

		if (requiredComponents.includes('processor')) {
			if (modelOpts.processor) {
				const { modelId, branch } = parseModelUrl(modelOpts.processor.url)
				const processorDownload = AutoProcessor.from_pretrained(modelId, {
					revision: branch,
					progress_callback: progressCallback,
				})
				downloadPromises.processor = processorDownload
			} else {
				const [hasProcessor, hasPreprocessor] = await Promise.all([
					remoteFileExists(
						`${config.url}/blob/${branch}/processor_config.json`,
					),
					remoteFileExists(
						`${config.url}/blob/${branch}/preprocessor_config.json`,
					),
				])
				if (hasProcessor || hasPreprocessor) {
					const processorDownload = AutoProcessor.from_pretrained(modelId, {
						revision: branch,
						progress_callback: progressCallback,
						// use_external_data_format: true,
					})
					downloadPromises.processor = processorDownload
				}
			}
		}
		await Promise.all(Object.values(downloadPromises))
		const modelComponents: TransformersJsModelComponents = {}
		if (downloadPromises.model) {
			modelComponents.model = (await downloadPromises.model) as PreTrainedModel
		}
		if (downloadPromises.tokenizer) {
			modelComponents.tokenizer =
				(await downloadPromises.tokenizer) as PreTrainedTokenizer
		}
		if (downloadPromises.processor) {
			modelComponents.processor =
				(await downloadPromises.processor) as Processor
		}
		return modelComponents
	}

	const downloadModel = async (validationResult: ModelFileValidationResult) => {
		log(
			LogLevels.info,
			`${validationResult.message} - Downloading model files`,
			{
				model: config.id,
				url: config.url,
				location: config.location,
				errors: validationResult.errors,
			},
		)
		const modelDownloadPromises = []
		const noModelConfigured =
			!config.textModel && !config.visionModel && !config.speechModel
		if (config.textModel || noModelConfigured) {
			const requiredComponents = validationResult.errors?.textModel
				? Object.keys(validationResult.errors.textModel)
				: undefined
			modelDownloadPromises.push(
				downloadModelFiles(config.textModel || {}, requiredComponents),
			)
		}
		if (config.visionModel) {
			const requiredComponents = validationResult.errors?.visionModel
				? Object.keys(validationResult.errors.visionModel)
				: undefined
			modelDownloadPromises.push(
				downloadModelFiles(config.visionModel, requiredComponents),
			)
		}
		if (config.speechModel) {
			const requiredComponents = validationResult.errors?.speechModel
				? Object.keys(validationResult.errors.speechModel)
				: undefined
			modelDownloadPromises.push(
				downloadModelFiles(config.speechModel, requiredComponents),
			)
		}
		const models = await Promise.all(modelDownloadPromises)
		for (const modelComponents of models) {
			disposeModelComponents(modelComponents)
		}
		if (signal?.aborted) {
			return
		}
		const modelCacheDir = path.join(env.cacheDir, modelId)
		await copyDirectory(modelCacheDir, config.location)
	}

	try {
		const validationResults = await validateModelFiles(config)
		if (signal?.aborted) {
			return
		}
		if (validationResults) {
			if (config.url) {
				await downloadModel(validationResults)
			} else {
				throw new Error(`Model files are invalid: ${validationResults.message}`)
			}
		}
	} catch (err) {
		clearFileLock()
		throw err
	}

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

	const modelLoadPromises = []
	const noModelConfigured =
		!config.textModel && !config.visionModel && !config.speechModel

	if (config.textModel || noModelConfigured) {
		modelLoadPromises.push(
			loadModelComponents(config.textModel || {}, config, modelPath),
		)
	} else {
		modelLoadPromises.push(Promise.resolve(undefined))
	}
	if (config.visionModel) {
		modelLoadPromises.push(
			loadModelComponents(config.visionModel, config, modelPath),
		)
	} else {
		modelLoadPromises.push(Promise.resolve(undefined))
	}
	if (config.speechModel) {
		modelLoadPromises.push(
			loadModelComponents(config.speechModel, config, modelPath),
		)
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
	const disposePromises = []
	if (instance.textModel) {
		disposePromises.push(disposeModelComponents(instance.textModel))
	}
	if (instance.visionModel) {
		disposePromises.push(disposeModelComponents(instance.visionModel))
	}
	if (instance.speechModel) {
		disposePromises.push(disposeModelComponents(instance.speechModel))
	}
	await Promise.all(disposePromises)
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
	if (!(instance.textModel?.tokenizer && instance.textModel?.model)) {
		throw new Error('Text model is not loaded.')
	}
	const inputTokens = instance.textModel.tokenizer(request.prompt)
	const outputTokens = await instance.textModel.model.generate({
		...inputTokens,
		max_new_tokens: request.maxTokens ?? 128,
	})
	// @ts-ignore
	const outputText = instance.textModel.tokenizer.batch_decode(outputTokens, {
		skip_special_tokens: true,
	})

	return {
		finishReason: 'eogToken',
		text: outputText[0],
		promptTokens: inputTokens.length,
		// @ts-ignore
		completionTokens: outputTokens.length,
		// @ts-ignore
		contextTokens: inputTokens.length + outputTokens.length,
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
			if (!instance.textModel?.tokenizer || !instance.textModel?.model) {
				throw new Error('Text model is not loaded.')
			}
			modelInputs = instance.textModel.tokenizer(embeddingInput.content, {
				padding: true, // pads input if it is shorter than context window
				truncation: true, // truncates input if it exceeds context window
			})
			inputTokens += modelInputs.input_ids.size
			const modelOutputs = await instance.textModel.model(modelInputs)
			result =
				modelOutputs.last_hidden_state ??
				modelOutputs.logits ??
				modelOutputs.token_embeddings ??
				modelOutputs.text_embeds
		} else if (embeddingInput.type === 'image') {
			if (!instance.visionModel?.processor || !instance.visionModel?.model) {
				throw new Error('Vision model is not loaded.')
			}
			const { data, info } = await embeddingInput.content.handle
				.raw()
				.toBuffer({ resolveWithObject: true })
			const image = new RawImage(
				new Uint8ClampedArray(data),
				info.width,
				info.height,
				info.channels,
			)
			modelInputs = await instance.visionModel.processor!(image)
			const modelOutputs = await instance.visionModel.model(modelInputs)
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
	if (!request.image) {
		throw new Error('No image provided')
	}
	const { data, info } = await request.image.handle
		.raw()
		.toBuffer({ resolveWithObject: true })
	const image = new RawImage(
		new Uint8ClampedArray(data),
		info.width,
		info.height,
		info.channels,
	)

	if (signal?.aborted) {
		return
	}

	const model = instance.visionModel || instance.textModel
	if (!(model && model.tokenizer && model.processor && model.model)) {
		throw new Error('No model loaded')
	}
	let textInputs = {}
	if (request.prompt) {
		textInputs = model!.tokenizer(request.prompt)
	}
	const imageInputs = await model.processor(image)
	const outputTokens = await model.model.generate({
		...textInputs,
		...imageInputs,
		max_new_tokens: request.maxTokens ?? 128,
	})
	// @ts-ignore
	const outputText = model.tokenizer.batch_decode(outputTokens, {
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
	if (!(instance.speechModel?.tokenizer && instance.speechModel?.model)) {
		throw new Error('No speech model loaded')
	}
	const streamer = new TextStreamer(instance.speechModel.tokenizer, {
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
		inputs = await instance.speechModel.processor!(audio)
	}

	const outputs = await instance.speechModel.model.generate({
		...inputs,
		max_new_tokens: request.maxTokens ?? 128,
		language: request.language ?? 'en',
		streamer,
	})

	// @ts-ignore
	const outputText = instance.speechModel.tokenizer.batch_decode(outputs, {
		skip_special_tokens: true,
	})

	return {
		text: outputText[0],
	}
}
