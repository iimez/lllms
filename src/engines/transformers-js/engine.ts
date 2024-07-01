import path from 'node:path'
import fs from 'node:fs'
import os from 'node:os'
import {
	EngineChatCompletionResult,
	EngineTextCompletionResult,
	EngineTextCompletionArgs,
	EngineChatCompletionArgs,
	EngineContext,
	EngineOptionsBase,
	ToolDefinition,
	ToolCallResultMessage,
	AssistantMessage,
	EngineEmbeddingArgs,
	EngineEmbeddingResult,
	CompletionFinishReason,
	ChatMessage,
	FileDownloadProgress,
	ModelConfig,
	EngineImageToTextArgs,
} from '#lllms/types/index.js'

import {
	env,
	PreTrainedModel,
	AutoProcessor,
	AutoTokenizer,
	RawImage,
	// @ts-ignore
} from '@xenova/transformers'
import { LogLevels } from '#lllms/lib/logger.js'
import { acquireFileLock } from '#lllms/lib/acquireFileLock.js'

export interface TransformersJsEngineOptions extends EngineOptionsBase {
	modelClass?: any // TODO type this; PreTrainedModel?
	dtype: any
}

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

function checkModelExists(config: ModelConfig<TransformersJsEngineOptions>) {
	if (!fs.existsSync(config.location)) {
		return false
	}
	if (!fs.existsSync(config.location + '/onnx')) {
		return false
	}
	if (config.engineOptions?.dtype) {
		if (typeof config.engineOptions?.dtype === 'string') {
			if (config.engineOptions?.dtype === 'fp32') {
				const expectedFile = `${config.location}/onnx/encoder_model.onnx`
				if (!fs.existsSync(expectedFile)) {
					console.debug('missing', expectedFile)
					return false
				}
			}
		} else if (typeof config.engineOptions?.dtype === 'object') {
			for (const fileName in config.engineOptions?.dtype) {
				const dataType = config.engineOptions?.dtype[fileName]
				const expectedFile = `${config.location}/onnx/${fileName}_${dataType}.onnx`
				if (!fs.existsSync(expectedFile)) {
					console.debug('missing', expectedFile)
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
	}: EngineContext<TransformersJsModelMeta, TransformersJsEngineOptions>,
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
	const ModelClass = config.engineOptions?.modelClass ?? PreTrainedModel
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
			dtype: config.engineOptions?.dtype,
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
		await Promise.all([modelDownload, processorDownload, tokenizerDownload])
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
			// console.debug({
			// 	source: sourceFile,
			// 	target: targetFile,
			// })
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
	}: EngineContext<TransformersJsModelMeta, TransformersJsEngineOptions>,
	signal?: AbortSignal,
) {
	const ModelClass = config.engineOptions?.modelClass ?? PreTrainedModel
	let modelPath = config.location
	if (!modelPath.endsWith('/')) {
		modelPath += '/'
	}
	// console.debug({
	// 	modelPath,
	// })

	// const device = config.engineOptions?.gpu ? 'cuda' : 'cpu'
	const modelPromise = ModelClass.from_pretrained(modelPath, {
		// dtype: 'fp32',
		local_files_only: true,
		cache_dir: '/',
		dtype: {
			embed_tokens: 'fp16',
			vision_encoder: 'fp32',
			encoder_model: 'fp16',
			decoder_model_merged: 'q4',
		},
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

	return {
		model,
		processor,
		tokenizer,
	}
}

export async function disposeInstance(instance: TransformersJsInstance) {
	instance.model.dispose()
}

// https://github.com/xenova/transformers.js/issues/815
// https://huggingface.co/microsoft/Florence-2-large-ft
// https://huggingface.co/onnx-community/Florence-2-large-ft/tree/main
// https://github.com/xenova/transformers.js/pull/545#issuecomment-2183625876
export async function processImageToTextTask(
	{ request }: EngineImageToTextArgs<TransformersJsEngineOptions>,
	instance: TransformersJsInstance,
	{ config, log }: EngineContext<TransformersJsEngineOptions>,
) {
	console.debug('processImageToTextTask', request)

	let image: any
	if (request.url) {
		image = await RawImage.fromURL(request.url)
	} else if (request.file) {
		const fileBuffer = fs.readFileSync(request.file)
		const blob = new Blob([fileBuffer])
		image = await RawImage.fromBlob(blob)
	}

	// Process inputs
	const prompts = 'Describe with a paragraph what is shown in the image.'
	const textInputs = instance.tokenizer(prompts)
	const visionInputs = await instance.processor(image)
	const outputTokens = await instance.model.generate({
		...textInputs,
		...visionInputs,
		max_new_tokens: 100,
	})
	const outputText = instance.tokenizer.batch_decode(outputTokens, {
		skip_special_tokens: true,
	})
	// console.debug(generatedText)

	return {
		text: outputText[0],
	}
}
