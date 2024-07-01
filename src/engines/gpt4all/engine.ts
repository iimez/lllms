import path from 'node:path'
import fs from 'node:fs'
import {
	loadModel,
	createCompletion,
	createEmbedding,
	InferenceModel,
	LoadModelOptions,
	CompletionInput,
	EmbeddingModel,
	DEFAULT_MODEL_LIST_URL,
} from 'gpt4all'
import {
	EngineTextCompletionArgs,
	EngineChatCompletionArgs,
	EngineChatCompletionResult,
	EngineTextCompletionResult,
	CompletionFinishReason,
	EngineContext,
	EngineOptionsBase,
	EngineEmbeddingArgs,
	EngineEmbeddingResult,
	FileDownloadProgress,
	ModelConfig,
} from '#lllms/types/index.js'
import { LogLevels } from '#lllms/lib/logger.js'
import { calculateFileChecksum } from '#lllms/lib/calculateFileChecksum.js'
import { downloadLargeFile } from '#lllms/lib/downloadLargeFile.js'
import { acquireFileLock } from '#lllms/lib/acquireFileLock.js'
import { createChatMessageArray } from './util.js'

export interface GPT4AllEngineOptions extends EngineOptionsBase {}
export type GPT4AllInstance = InferenceModel | EmbeddingModel

export interface GPT4AllModelMeta {
	url: string
	md5sum: string
	filename: string
	promptTemplate: string
	systemPrompt: string
	filesize: number
	ramrequired: number
}

// interface GPT4AllModelConfig {
// }

function checkModelExists(config: ModelConfig<GPT4AllEngineOptions>) {
	if (!fs.existsSync(config.location)) {
		return false
	}
	return true
}

async function verifyModelFile(location: string, md5: string) {
	const fileHash = await calculateFileChecksum(location, 'md5')
	if (fileHash !== md5) {
		throw new Error(
			`Model md5 checksum mismatch: expected ${md5} got ${fileHash} for ${location}`,
		)
	}
}

export const autoGpu = true

export async function prepareModel(
	{ config, log }: EngineContext<GPT4AllModelMeta, GPT4AllEngineOptions>,
	onProgress?: (progress: FileDownloadProgress) => void,
	signal?: AbortSignal,
) {
	fs.mkdirSync(path.dirname(config.location), { recursive: true })
	const clearFileLock = await acquireFileLock(config.location, signal)
	if (signal?.aborted) {
		return
	}
	log(LogLevels.info, `Preparing gpt4all model at ${config.location}`, {
		model: config.id,
	})
	let modelMeta: GPT4AllModelMeta | undefined
	let modelList: GPT4AllModelMeta[]
	const modelMetaPath = path.join(path.dirname(config.location), 'models.json')
	if (!fs.existsSync(modelMetaPath)) {
		const res = await fetch(DEFAULT_MODEL_LIST_URL)
		modelList = (await res.json()) as GPT4AllModelMeta[]
		fs.writeFileSync(modelMetaPath, JSON.stringify(modelList, null, 2))
	} else {
		modelList = JSON.parse(fs.readFileSync(modelMetaPath, 'utf-8'))
	}
	const foundModelMeta = modelList.find(
		(item) => {
			if (config.md5 && item.md5sum) {
				return item.md5sum === config.md5
			}
			if (config.url && item.url) {
				return item.url === config.url
			}
			return item.filename === path.basename(config.location)
		},
	)
	if (foundModelMeta) {
		modelMeta = foundModelMeta
	}

	if (!checkModelExists(config)) {
		if (!config.url) {
			throw new Error(`Cannot download "${config.id}" - no URL configured`)
		}
		if (signal?.aborted) {
			return
		}
		await downloadLargeFile({
			url: config.url,
			file: config.location,
			onProgress,
			signal,
		})
	}

	if (!signal?.aborted) {
		if (config.md5) {
			await verifyModelFile(config.location, config.md5)
		} else if (modelMeta?.md5sum) {
			await verifyModelFile(config.location, modelMeta.md5sum)
		}
	}
	clearFileLock()
	return modelMeta
}

export async function createInstance(
	{ config, log }: EngineContext<GPT4AllModelMeta, GPT4AllEngineOptions>,
	signal?: AbortSignal,
) {
	log(LogLevels.info, `Load GPT4All model ${config.location}`)
	const loadOpts: LoadModelOptions = {
		modelPath: path.dirname(config.location),
		// file: config.file,
		modelConfigFile: path.dirname(config.location) + '/models.json',
		allowDownload: false,
		device: config.engineOptions?.gpu ? 'gpu' : 'cpu',
		ngl: config.engineOptions?.gpuLayers ?? 100,
		nCtx: config.contextSize ?? 2048,
		// verbose: true,
		// signal?: // TODO no way to cancel load
	}

	let modelType: 'inference' | 'embedding'
	if (config.task === 'text-completion') {
		modelType = 'inference'
	} else if (config.task === 'embedding') {
		modelType = 'embedding'
	} else {
		throw new Error(`Unsupported task type: ${config.task}`)
	}

	const instance = await loadModel(path.basename(config.location), {
		...loadOpts,
		type: modelType,
	})
	if (config.engineOptions?.cpuThreads) {
		instance.llm.setThreadCount(config.engineOptions.cpuThreads)
	}

	if (config.preload && 'generate' in instance) {
		if ('messages' in config.preload) {
			let messages = createChatMessageArray(config.preload.messages)
			let systemPrompt
			if (messages[0].role === 'system') {
				systemPrompt = messages[0].content
				messages = messages.slice(1)
			}
			await instance.createChatSession({
				systemPrompt,
				messages,
			})
		} else if ('prefix' in config.preload) {
			await instance.generate(config.preload.prefix, {
				nPredict: 0,
			})
		} else {
			await instance.generate('', {
				nPredict: 0,
			})
		}
	}

	return instance
}

export async function disposeInstance(instance: GPT4AllInstance) {
	return instance.dispose()
}

export async function processTextCompletionTask(
	{ request, config, onChunk }: EngineTextCompletionArgs<GPT4AllEngineOptions>,
	instance: GPT4AllInstance,
	signal?: AbortSignal,
): Promise<EngineTextCompletionResult> {
	if (!('generate' in instance)) {
		throw new Error('Instance does not support text completion.')
	}
	if (!request.prompt) {
		throw new Error('Prompt is required for text completion.')
	}

	let finishReason: CompletionFinishReason = 'eogToken'
	let suffixToRemove: string | undefined

	const defaults = config.completionDefaults ?? {}
	const stopTriggers = request.stop ?? defaults.stop ?? []
	const includesStopTriggers = (text: string) =>
		stopTriggers.find((t) => text.includes(t))
	const result = await instance.generate(request.prompt, {
		// @ts-ignore
		special: true, // allows passing in raw prompt (including <|start|> etc.)
		promptTemplate: '%1',
		temperature: request.temperature ?? defaults.temperature,
		nPredict: request.maxTokens ?? defaults.maxTokens,
		topP: request.topP ?? defaults.topP,
		topK: request.topK ?? defaults.topK,
		minP: request.minP ?? defaults.minP,
		nBatch: config.engineOptions?.batchSize,
		// TODO not sure if repeatPenalty interacts with repeatLastN and how it differs from frequency/presencePenalty.
		repeatLastN: request.repeatPenaltyNum ?? defaults.repeatPenaltyNum,
		repeatPenalty: request.repeatPenalty ?? defaults.repeatPenalty,
		// seed: args.seed, // https://github.com/nomic-ai/gpt4all/issues/1952
		onResponseToken: (tokenId, text) => {
			const matchingTrigger = includesStopTriggers(text)
			if (matchingTrigger) {
				finishReason = 'stopTrigger'
				suffixToRemove = text
				return false
			}
			if (onChunk) {
				onChunk({
					text,
					tokens: [tokenId],
				})
			}
			return !signal?.aborted
		},
		// @ts-ignore
		onResponseTokens: ({ tokenIds, text }) => {
			const matchingTrigger = includesStopTriggers(text)
			if (matchingTrigger) {
				finishReason = 'stopTrigger'
				suffixToRemove = text
				return false
			}
			if (onChunk) {
				onChunk({
					text,
					tokens: tokenIds,
				})
			}
			return !signal?.aborted
		},
	})

	if (result.tokensGenerated === request.maxTokens) {
		finishReason = 'maxTokens'
	}

	let responseText = result.text
	if (suffixToRemove) {
		responseText = responseText.slice(0, -suffixToRemove.length)
	}

	return {
		finishReason,
		text: responseText,
		promptTokens: result.tokensIngested,
		completionTokens: result.tokensGenerated,
		totalTokens: result.tokensIngested + result.tokensGenerated,
	}
}

export async function processChatCompletionTask(
	{
		request,
		config,
		resetContext,
		onChunk,
	}: EngineChatCompletionArgs<GPT4AllEngineOptions>,
	instance: GPT4AllInstance,
	signal?: AbortSignal,
): Promise<EngineChatCompletionResult> {
	if (!('createChatSession' in instance)) {
		throw new Error('Instance does not support chat completion.')
	}
	let session = instance.activeChatSession
	if (!session || resetContext) {
		let messages = createChatMessageArray(request.messages)
		let systemPrompt
		if (messages[0].role === 'system') {
			systemPrompt = messages[0].content
			messages = messages.slice(1)
		}
		// drop last user message
		if (messages[messages.length - 1].role === 'user') {
			messages = messages.slice(0, -1)
		}

		session = await instance.createChatSession({
			systemPrompt,
			messages,
		})
	}

	const conversationMessages = createChatMessageArray(request.messages).filter(
		(m) => m.role !== 'system',
	)

	const lastMessage = conversationMessages[conversationMessages.length - 1]
	if (lastMessage.role !== 'user') {
		throw new Error('Last message must be from user.')
	}
	const input: CompletionInput = lastMessage.content

	let finishReason: CompletionFinishReason = 'eogToken'
	let suffixToRemove: string | undefined

	const defaults = config.completionDefaults ?? {}
	const stopTriggers = request.stop ?? defaults.stop ?? []
	const includesStopTriggers = (text: string) =>
		stopTriggers.find((t) => text.includes(t))
	const result = await createCompletion(session, input, {
		temperature: request.temperature ?? defaults.temperature,
		nPredict: request.maxTokens ?? defaults.maxTokens,
		topP: request.topP ?? defaults.topP,
		topK: request.topK ?? defaults.topK,
		minP: request.minP ?? defaults.minP,
		nBatch: config.engineOptions?.batchSize,
		repeatLastN: request.repeatPenaltyNum ?? defaults.repeatPenaltyNum,
		repeatPenalty: request.repeatPenalty ?? defaults.repeatPenalty,
		// seed: args.seed, // see https://github.com/nomic-ai/gpt4all/issues/1952
		onResponseToken: (tokenId, text) => {
			const matchingTrigger = includesStopTriggers(text)
			if (matchingTrigger) {
				finishReason = 'stopTrigger'
				suffixToRemove = text
				return false
			}
			if (onChunk) {
				onChunk({
					text,
					tokens: [tokenId],
				})
			}
			return !signal?.aborted
		},
		// @ts-ignore
		onResponseTokens: ({ tokenIds, text }) => {
			const matchingTrigger = includesStopTriggers(text)
			if (matchingTrigger) {
				finishReason = 'stopTrigger'
				suffixToRemove = text
				return false
			}
			if (onChunk) {
				onChunk({
					tokens: tokenIds,
					text,
				})
			}

			return !signal?.aborted
		},
	})

	if (result.usage.completion_tokens === request.maxTokens) {
		finishReason = 'maxTokens'
	}

	let response = result.choices[0].message.content
	if (suffixToRemove) {
		response = response.slice(0, -suffixToRemove.length)
	}

	return {
		finishReason,
		message: {
			role: 'assistant',
			content: response,
		},
		promptTokens: result.usage.prompt_tokens,
		completionTokens: result.usage.completion_tokens,
		totalTokens: result.usage.total_tokens,
	}
}

export async function processEmbeddingTask(
	{ request, config }: EngineEmbeddingArgs<GPT4AllEngineOptions>,
	instance: GPT4AllInstance,
	signal?: AbortSignal,
): Promise<EngineEmbeddingResult> {
	if (!('embed' in instance)) {
		throw new Error('Instance does not support embedding.')
	}
	if (!request.input) {
		throw new Error('Input is required for embedding.')
	}
	const texts: string[] = []
	if (typeof request.input === 'string') {
		texts.push(request.input)
	} else {
		const strInputs = request.input.filter(
			(i) => typeof i === 'string',
		) as string[]
		texts.push(...strInputs)
	}

	const res = await createEmbedding(instance, texts, {
		dimensionality: request.dimensions,
	})

	return {
		embeddings: res.embeddings,
		inputTokens: res.n_prompt_tokens,
	}
}
