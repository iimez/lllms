import {
	getLlama,
	LlamaChatSession,
	LlamaModel,
	LlamaContext,
	LlamaCompletion,
	ChatWrapper,
	ChatHistoryItem,
	LlamaLogLevel,
	LlamaText,
	resolveChatWrapper,
	TokenBias,
	Token,
} from 'node-llama-cpp'
import { StopGenerationTrigger } from 'node-llama-cpp/dist/utils/StopGenerationDetector.js'
import {
	EngineChatCompletionResult,
	EngineCompletionResult,
	EngineCompletionContext,
	EngineChatCompletionContext,
	EngineContext,
	EngineOptionsBase,
} from '#lllms/types/index.js'
import { LogLevels } from '#lllms/lib/logger.js'
import { formatBytes, mergeAbortSignals } from '#lllms/lib/util.js'

// https://github.com/withcatai/node-llama-cpp/pull/105
// https://github.com/withcatai/node-llama-cpp/discussions/109

export interface LlamaCppOptions extends EngineOptionsBase {
	memLock?: boolean
}

interface LlamaCppInstance {
	model: LlamaModel
	context: LlamaContext
	session: LlamaChatSession
}

export async function loadInstance(
	{ config, log }: EngineContext<LlamaCppOptions>,
	signal?: AbortSignal,
) {
	log(LogLevels.debug, 'Load Llama model', config.engineOptions)

	const llama = await getLlama({
		// may be "auto" | "metal" | "cuda" | "vulkan"
		gpu: config.engineOptions?.gpu ? 'auto' : false,
		// forwarding llama logger
		logLevel: LlamaLogLevel.debug,
		logger: (level, message) => {
			if (level === LlamaLogLevel.warn) {
				log(LogLevels.warn, message)
			} else if (
				level === LlamaLogLevel.error ||
				level === LlamaLogLevel.fatal
			) {
				log(LogLevels.error, message)
			} else if (
				level === LlamaLogLevel.info ||
				level === LlamaLogLevel.debug
			) {
				log(LogLevels.verbose, message)
			}
		},
	})
	const model = await llama.loadModel({
		modelPath: config.file, // full model absolute path
		loadSignal: signal,
		useMlock: config.engineOptions?.memLock ?? false,
		gpuLayers: config.engineOptions?.gpuLayers,
		// onLoadProgress: (percent) => {}
	})

	const context = await model.createContext({
		sequences: 2,
		seed: createSeed(0, 1000000),
		threads: config.engineOptions?.cpuThreads,
		batchSize: config.engineOptions?.batchSize,
		contextSize: config.contextSize,
		// batching: {
		// 	dispatchSchedule: 'nextTick',
		// 	itemPrioritizationStrategy: 'maximumParallelism',
		// 	itemPrioritizationStrategy: 'firstInFirstOut',
		// },
		createSignal: signal,
	})

	return {
		model,
		context,
		session: null,
	}
}

export async function disposeInstance(instance: LlamaCppInstance) {
	instance.model.dispose()
}

function createSeed(min: number, max: number) {
	min = Math.ceil(min)
	max = Math.floor(max)
	return Math.floor(Math.random() * (max - min)) + min
}

interface ChatWrapperWithStopGenerationTrigger extends ChatWrapper {
	stopGenerationTriggers: string[] | null
	setExtraStopGenerationTriggers(triggers: string[] | null): void
}

// this extends whatever chatwrapper is passed in to support setting (and unsetting) stop generation triggers
function withCustomStopGenerationTrigger(
	chatWrapper: ChatWrapper,
): ChatWrapperWithStopGenerationTrigger {
	const customChatWrapper = chatWrapper as ChatWrapperWithStopGenerationTrigger
	customChatWrapper.setExtraStopGenerationTriggers = (
		triggers: string[] | null,
	) => {
		customChatWrapper.stopGenerationTriggers = triggers
	}
	customChatWrapper.stopGenerationTriggers = null
	const generateContextText = chatWrapper.generateContextText.bind(chatWrapper)
	customChatWrapper.generateContextText = (history, options) => {
		const result = generateContextText(history, options)
		if (customChatWrapper.stopGenerationTriggers) {
			const extraStopGenerationTriggers =
				customChatWrapper.stopGenerationTriggers.map((s) => LlamaText(s))
			result.stopGenerationTriggers.push(...extraStopGenerationTriggers)
		}
		return result
	}
	return customChatWrapper
}

export async function processChatCompletion(
	instance: LlamaCppInstance,
	{
		request,
		resetContext,
		log,
		onChunk,
	}: EngineChatCompletionContext<LlamaCppOptions>,
	signal?: AbortSignal,
): Promise<EngineChatCompletionResult> {
	const nonSystemMessages = request.messages.filter((m) => m.role !== 'system')

	if (!instance.session || !instance.session.context.sequencesLeft) {
		if (instance.session && !instance.session?.context?.sequencesLeft) {
			log(LogLevels.debug, 'No sequencesLeft, recreating LlamaChatSession')
		}
		if (!instance.session) {
			log(LogLevels.debug, 'Creating LlamaChatSession')
		}
		if (instance.session) {
			await instance.session.dispose()
		}
		const model = instance.model
		const chatWrapper = resolveChatWrapper({
			type: 'auto',
			bosString: model.tokens.bosString,
			filename: model.filename,
			fileInfo: model.fileInfo,
			tokenizer: model.tokenizer,
		})
		let systemPrompt = request.systemPrompt
		if (!systemPrompt && request.messages[0].role === 'system') {
			systemPrompt = request.messages[0].content
		}
		instance.session = new LlamaChatSession({
			systemPrompt,
			chatWrapper: withCustomStopGenerationTrigger(chatWrapper),
			contextSequence: instance.context.getSequence(),
			// contextShift: {
			// 	size: 50,
			// 	strategy: 'eraseFirstResponseAndKeepFirstSystem',
			// },
		})
	}

	if (resetContext) {
		const conversationMessages: ChatHistoryItem[] = []
		for (const message of nonSystemMessages) {
			conversationMessages.push({
				type: message.role,
				text: message.content,
			} as ChatHistoryItem)
		}
		// drop last user message, thats what we wanna prompt with.
		if (conversationMessages[conversationMessages.length - 1].type === 'user') {
			conversationMessages.pop()
		}
		instance.session.setChatHistory(conversationMessages)
	}

	// set additional stop generation triggers for this completion
	if (request.stop?.length) {
		// @ts-ignore
		instance.session.chatWrapper.setExtraStopGenerationTriggers(request.stop)
	} else {
		// @ts-ignore
		instance.session.chatWrapper.setExtraStopGenerationTriggers(null)
	}

	// setting up logit/token bias.
	let tokenBias: TokenBias | undefined
	if (request.tokenBias) {
		tokenBias = new TokenBias(instance.model)
		for (const key in request.tokenBias) {
			const bias = request.tokenBias[key] / 10
			const tokenId = parseInt(key) as Token
			if (!isNaN(tokenId)) {
				tokenBias.set(tokenId, bias)
			} else {
				tokenBias.set(key, bias)
			}
		}
	}

	// what goes into promptWithMeta is the last user message.
	const lastMessage = nonSystemMessages[nonSystemMessages.length - 1]
	if (lastMessage.role !== 'user') {
		throw new Error('Last message must be from user.')
	}
	const input = lastMessage.content
	let inputTokens = instance.model.tokenize(input)
	let generatedTokenCount = 0
	let completionResult: EngineChatCompletionResult
	let partialResponse = ''

	try {
		const result = await instance.session.promptWithMeta(input, {
			maxTokens: request.maxTokens,
			temperature: request.temperature,
			topP: request.topP,
			topK: request.topK,
			minP: request.minP,
			tokenBias,
			repeatPenalty: {
				lastTokens: request.repeatPenaltyNum ?? 64,
				frequencyPenalty: request.frequencyPenalty,
				presencePenalty: request.presencePenalty,
			},
			signal,
			onToken: (tokens) => {
				generatedTokenCount += tokens.length
				const text = instance.model.detokenize(tokens)
				partialResponse += text
				if (onChunk) {
					onChunk({
						tokens,
						text,
					})
				}
			},
		})

		completionResult = {
			finishReason: result.stopReason,
			message: {
				role: 'assistant',
				content: result.responseText,
			},
			promptTokens: inputTokens.length,
			completionTokens: generatedTokenCount,
			totalTokens: inputTokens.length + generatedTokenCount,
		}
	} catch (error: any) {
		if (error.name !== 'AbortError') {
			throw error
		} else {
			// if the completion was aborted, return the partial response
			completionResult = {
				finishReason: 'cancelled',
				message: {
					role: 'assistant',
					content: partialResponse,
				},
				promptTokens: inputTokens.length,
				completionTokens: generatedTokenCount,
				totalTokens: inputTokens.length + generatedTokenCount,
			}
		}
	}

	const ingestedMessages =
		instance.session.getLastEvaluationContextWindow() ?? []
	console.debug('State after completion', {
		stateSize: formatBytes(instance.context.stateSize),
		sequencesLeft: instance.context.sequencesLeft,
		tokenCount: instance.session.sequence.contextTokens.length,
		incomingMessageCount: request.messages.length,
		ingestedMessageCount: ingestedMessages.length,
	})

	return completionResult
}

export async function processCompletion(
	instance: LlamaCppInstance,
	{ request, config, log, onChunk }: EngineCompletionContext<LlamaCppOptions>,
	signal?: AbortSignal,
): Promise<EngineCompletionResult> {
	if (!request.prompt) {
		throw new Error('Prompt is required for completion.')
	}

	if (instance.context.sequencesLeft) {
		log(LogLevels.debug, 'Clearing history', {
			sequencesLeft: instance.context.sequencesLeft,
		})
		await instance.context.getSequence().clearHistory()
	} else {
		log(LogLevels.debug, 'No sequencesLeft, recreating context')
		await instance.context.dispose()
		instance.context = await instance.model.createContext({
			createSignal: signal,
			seed: request.seed, // || createSeed(0, 1000000),
			threads: config.engineOptions?.cpuThreads,
			batchSize: config.engineOptions?.batchSize,
		})
	}

	const completion = new LlamaCompletion({
		contextSequence: instance.context.getSequence(),
	})

	const stopTriggers: StopGenerationTrigger = []
	if (request.stop) {
		stopTriggers.push(...request.stop)
	}

	const tokens = instance.model.tokenize(request.prompt)
	let generatedTokenCount = 0
	const result = await completion.generateCompletionWithMeta(tokens, {
		maxTokens: request.maxTokens,
		temperature: request.temperature,
		topP: request.topP,
		topK: request.topK,
		repeatPenalty: {
			lastTokens: request.repeatPenaltyNum ?? 64,
			frequencyPenalty: request.frequencyPenalty,
			presencePenalty: request.presencePenalty,
		},
		signal: signal,
		stopGenerationTriggers: stopTriggers.length ? [stopTriggers] : undefined,
		onToken: (tokens) => {
			generatedTokenCount += tokens.length
			const text = instance.model.detokenize(tokens)
			if (onChunk) {
				onChunk({
					tokens,
					text,
				})
			}
		},
	})

	completion.dispose()

	return {
		finishReason: result.metadata.stopReason,
		text: result.response,
		promptTokens: tokens.length,
		completionTokens: generatedTokenCount,
		totalTokens: tokens.length + generatedTokenCount,
	}
}
