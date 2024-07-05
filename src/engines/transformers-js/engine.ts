import path from 'node:path'
import fs from 'node:fs'

import {
	EngineContext,
	FileDownloadProgress,
	ModelConfig,
	EngineImageToTextArgs,
	EngineSpeechToTextArgs,
} from '#lllms/types/index.js'

import {
	env,
	PreTrainedModel,
	AutoProcessor,
	AutoTokenizer,
	RawImage,
	TextStreamer,
	// @ts-ignore
} from '@xenova/transformers'
import { LogLevels } from '#lllms/lib/logger.js'
import { acquireFileLock } from '#lllms/lib/acquireFileLock.js'
import { decodeAudio } from '#lllms/lib/audio.js'

// TODO transformers.js types currently hard to fix, until v3 is released on npm (with typedefs)

interface TransformersJsInstance {
	model: any
	processor: any
	tokenizer: any
}

interface ModelFile {
	file: string
	size: number
}

interface TransformersJsModelMeta {
	modelType: string
	files: ModelFile[]
}

export interface TransformersJsModelConfig extends ModelConfig {
	location: string
	url: string
	modelClass?: any
	dtype?: Record<string, string> | string
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
	if (config.dtype) {
		if (typeof config.dtype === 'string') {
			if (config.dtype === 'fp32') {
				const expectedFile = `${config.location}/onnx/encoder_model.onnx`
				if (!fs.existsSync(expectedFile)) {
					console.debug('missing', config.dtype, expectedFile)
					return false
				}
			}
		} else if (typeof config.dtype === 'object') {
			for (const fileName in config.dtype) {
				const dataType = config.dtype[fileName]
				let expectedFile = `${config.location}/onnx/${fileName}_${dataType}.onnx`
				if (dataType === 'fp32') {
					expectedFile = `${config.location}/onnx/${fileName}.onnx`
				}
				if (!fs.existsSync(expectedFile)) {
					console.debug('missing', config.dtype, expectedFile)
					return false
				}
			}
		}
	}
	return true
}

export const autoGpu = true

export async function prepareModel(
	{
		config,
		log,
	}: EngineContext<TransformersJsModelConfig>,
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
	// TODO support other hostnames than hugingface.co?
	const urlSegments = parsedUrl.pathname.split('/')
	const org = urlSegments[1]
	const repo = urlSegments[2]
	const branch = urlSegments[4] || 'main'

	const modelId = `${org}/${repo}`
	const ModelClass = config.modelClass ?? PreTrainedModel
	const modelFiles: ModelFile[] = []
	const configMeta: any = {}

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
		const modelDownload = ModelClass.from_pretrained(modelId, {
			revision: branch,
			dtype: config.dtype,
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
		const processorDownload = AutoProcessor.from_pretrained(modelId)
		const tokenizerDownload = AutoTokenizer.from_pretrained(modelId)
		const [model, ] = await Promise.all([modelDownload, processorDownload, tokenizerDownload])
		model.dispose()
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
	{
		config,
		log,
	}: EngineContext<TransformersJsModelConfig>,
	signal?: AbortSignal,
) {
	const ModelClass = config.modelClass ?? PreTrainedModel
	let modelPath = config.location
	if (!modelPath.endsWith('/')) {
		modelPath += '/'
	}
	const modelPromise = ModelClass.from_pretrained(modelPath, {
		local_files_only: true,
		cache_dir: '/',
		dtype: config.dtype,
		device: config.device?.gpu ? 'gpu' : 'cpu',
	})
	const processorPromise = AutoProcessor.from_pretrained(modelPath, {
		local_files_only: true,
		cache_dir: '/',
	})
	const tokenizerPromise = AutoTokenizer.from_pretrained(modelPath, {
		local_files_only: true,
		cache_dir: '/',
	})

	const [model, processor, tokenizer] = await Promise.all([
		modelPromise,
		processorPromise,
		tokenizerPromise,
	])

	// TODO preload whisper / any speech to text?
	// await model.generate({
	// 	input_features: full([1, 80, 3000], 0.0),
	// 	max_new_tokens: 1,
	// });

	return {
		model,
		processor,
		tokenizer,
	}
}

export async function disposeInstance(instance: TransformersJsInstance) {
	instance.model.dispose()
}

export async function processImageToTextTask(
	{ request, config, log }: EngineImageToTextArgs,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
) {
	let image: any
	if (request.url) {
		image = await RawImage.fromURL(request.url)
	} else if (request.file) {
		const fileBuffer = fs.readFileSync(request.file)
		const blob = new Blob([fileBuffer])
		image = await RawImage.fromBlob(blob)
	}

	// Process inputs
	let textInputs = {}
	if (request.prompt) {
		textInputs = instance.tokenizer(request.prompt)
	}
	const visionInputs = await instance.processor(image)
	const outputTokens = await instance.model.generate({
		...textInputs,
		...visionInputs,
		max_new_tokens: request.maxTokens ?? 128,
		
	})
	const outputText = instance.tokenizer.batch_decode(outputTokens, {
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

export async function processSpeechToTextTask(
	{ request, onChunk }: EngineSpeechToTextArgs,
	instance: TransformersJsInstance,
	signal?: AbortSignal,
) {
	const streamer = new TextStreamer(instance.tokenizer, {
		skip_prompt: true,
		skip_special_tokens: true,
		callback_function: (output: any) => {
			if (onChunk) {
				onChunk({ text: output })
			}
		},
	})
	let inputs
	if (request.file) {
		const audio = await readAudioFile(request.file)
		inputs = await instance.processor(audio)
	}

	const outputs = await instance.model.generate({
		...inputs,
		max_new_tokens: request.maxTokens ?? 128,
		language: request.language ?? 'en',
		streamer,
	})

	const outputText = instance.tokenizer.batch_decode(outputs, {
		skip_special_tokens: true,
	})
	
	return {
		text: outputText[0],
	}
}
