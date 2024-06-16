import path from 'node:path'
import {
	loadModel,
	createCompletion,
	createEmbedding,
	InferenceModel,
	LoadModelOptions,
	CompletionInput,
	ChatMessage,
	EmbeddingModel,
} from 'gpt4all'
import {
	EngineCompletionContext,
	EngineChatCompletionContext,
	EngineChatCompletionResult,
	EngineCompletionResult,
	CompletionFinishReason,
	EngineContext,
	EngineOptionsBase,
	EngineEmbeddingContext,
	EngineEmbeddingsResult,
} from '#lllms/types/index.js'
import { LogLevels } from '#lllms/lib/logger.js'

export interface GPT4AllOptions extends EngineOptionsBase {}

export async function loadInstance(
	{ config, log }: EngineContext<GPT4AllOptions>,
	signal?: AbortSignal,
) {
	log(LogLevels.info, `Load GPT4All model ${config.file}`)

	const loadOpts: LoadModelOptions = {
		modelPath: path.dirname(config.file),
		// file: config.file,
		// allowDownload: false,
		device: config.engineOptions?.gpu ? 'gpu' : 'cpu',
		ngl: config.engineOptions?.gpuLayers ?? 100,
		nCtx: config.contextSize ?? 2048,
		// verbose: true,
		// signal?: // TODO no way to cancel load
	}

	const instance = await loadModel(path.basename(config.file), {
		...loadOpts,
		type: config.task,
	})
	if (config.engineOptions?.cpuThreads) {
		instance.llm.setThreadCount(config.engineOptions.cpuThreads)
	}
	return instance
}

export async function disposeInstance(instance: InferenceModel) {
	return instance.dispose()
}

export async function processCompletion(
	instance: InferenceModel,
	{ request, config, onChunk }: EngineCompletionContext<GPT4AllOptions>,
	signal?: AbortSignal,
): Promise<EngineCompletionResult> {
	if (!request.prompt) {
		throw new Error('Prompt is required for completion.')
	}

	let finishReason: CompletionFinishReason = 'eogToken'
	let suffixToRemove: string | undefined

	const defaults = config.completionDefaults ?? {}
	const stopTriggers = request.stop ?? defaults.stop ?? []
	const includesStopTriggers = (text: string) => stopTriggers.find((t) => text.includes(t))
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

export async function processChatCompletion(
	instance: InferenceModel,
	{
		request,
		config,
		resetContext,
		onChunk,
	}: EngineChatCompletionContext<GPT4AllOptions>,
	signal?: AbortSignal,
): Promise<EngineChatCompletionResult> {
	let session = instance.activeChatSession
	if (!session || resetContext) {
		let systemPrompt = request.systemPrompt ?? config.systemPrompt
		// allow overriding system prompt via initial message.
		if (request.messages[0].role === 'system') {
			systemPrompt = request.messages[0].content
		}
		session = await instance.createChatSession({
			systemPrompt,
		})
	}

	// if we have reset context, we need to reingest the chat history,
	// or otherwise just append the last user message.
	const nonSystemMessages = request.messages.filter(
		(m) => m.role !== 'system' && m.role !== 'function',
	)
	let input: CompletionInput

	if (resetContext) {
		// reingests all, then prompts automatically for last user message
		input = nonSystemMessages as ChatMessage[]
	} else {
		// append the last (user) message
		const lastMessage = nonSystemMessages[nonSystemMessages.length - 1]
		if (lastMessage.role !== 'user') {
			throw new Error('Last message must be from user.')
		}
		input = lastMessage.content
	}

	let finishReason: CompletionFinishReason = 'eogToken'
	let suffixToRemove: string | undefined

	const defaults = config.completionDefaults ?? {}
	const stopTriggers = request.stop ?? defaults.stop ?? []
	const includesStopTriggers = (text: string) => stopTriggers.find((t) => text.includes(t))
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
		onResponseTokens: ({tokenIds, text}) => {
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

export async function processEmbeddings(
	instance: EmbeddingModel,
	{ request, config }: EngineEmbeddingContext<GPT4AllOptions>,
	signal?: AbortSignal,
): Promise<EngineEmbeddingsResult> {
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
