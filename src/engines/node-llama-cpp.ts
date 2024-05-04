import {
	getLlama,
	LlamaChatSession,
	LlamaModel,
	LlamaContext,
	ChatMLChatWrapper,
	LlamaCompletion,
	Llama3ChatWrapper,
	GeneralChatWrapper,
	ChatWrapper,
	ChatHistoryItem,
	LlamaLogLevel,
	Llama,
	LlamaText,
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
import { PhiChatWrapper} from './PhiChatWrapper.js'

// https://github.com/withcatai/node-llama-cpp/pull/105
// https://github.com/withcatai/node-llama-cpp/discussions/109

export interface LlamaCppOptions extends EngineOptionsBase {
	memLock?: boolean
}

interface LlamaCppInstance {
	model: LlamaModel
	context: LlamaContext | null
	session: LlamaChatSession | null
}

function pickChatWrapper(
	templateFormat: string | undefined
): ChatWrapper | undefined {
	switch (templateFormat) {
		case 'chatml':
			return new ChatMLChatWrapper()
		case 'llama3':
			return new Llama3ChatWrapper()
		case 'phi':
			return new PhiChatWrapper()
	}
	return undefined
}

export async function loadInstance(
	{ id, config, log }: EngineContext<LlamaCppOptions>,
	signal?: AbortSignal,
) {
	log(LogLevels.debug, 'Load Llama model', {
		instance: id,
		...config.engineOptions,
	})
	
	const llama = await getLlama({
		// may be "auto" | "metal" | "cuda" | "vulkan"
		gpu: config.engineOptions?.gpu ? 'auto' : false,
		// forwarding logger
		logLevel: LlamaLogLevel.debug,
		logger: (level, message) => {
			if (level === LlamaLogLevel.warn) {
				log(LogLevels.warn, message, { instance: id })
			} else if (
				level === LlamaLogLevel.error ||
				level === LlamaLogLevel.fatal
			) {
				log(LogLevels.error, message, { instance: id })
			} else if (
				level === LlamaLogLevel.info ||
				level === LlamaLogLevel.debug
			) {
				log(LogLevels.verbose, message, { instance: id })
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

export async function processChatCompletion(
	instance: LlamaCppInstance,
	{
		config,
		request,
		resetContext,
		log,
		onChunk,
	}: EngineChatCompletionContext<LlamaCppOptions>,
	signal?: AbortSignal,
): Promise<EngineChatCompletionResult> {
	if (resetContext || !instance.session || !instance.context?.sequencesLeft) {
		// allow setting system prompt via initial message.
		let systemPrompt = request.systemPrompt
		if (!systemPrompt && request.messages[0].role === 'system') {
			systemPrompt = request.messages[0].content
		}

		if (!instance.context?.sequencesLeft) {
			log(LogLevels.warn, 'No sequences left, recreating context.')
			// TODO very unsure about this.
			if (instance.context) {
				await instance.context.dispose()
			}
			if (instance.session) {
				await instance.session.dispose({ disposeSequence: true })
				instance.session = null
			}
			instance.context = await instance.model.createContext({
				sequences: 2,
				createSignal: signal,
				seed: request.seed || createSeed(0, 1000000),
				threads: config.engineOptions?.cpuThreads,
				batchSize: config.engineOptions?.batchSize,
				contextSize: config.contextSize,
			})
		}
		
		if (!instance.session) {
			instance.session = new LlamaChatSession({
				chatWrapper: pickChatWrapper(config.templateFormat),
				systemPrompt,
				contextSequence: instance.context.getSequence(),
				autoDisposeSequence: true,
				// contextShift: {
				// 	size: 50,
				// 	strategy: "eraseFirstResponseAndKeepFirstSystem"
				// },
			})
		}
	}
	
	// TODO extend chatwrapper properly
	if ('setStopGenerationTriggers' in instance.session.chatWrapper) {
		if (request.stop?.length) {
			// @ts-ignore
			instance.session.chatWrapper.setStopGenerationTriggers(request.stop)
		} else {
			// @ts-ignore
			instance.session.chatWrapper.setStopGenerationTriggers(null)
		}
	}

	const nonSystemMessages = request.messages.filter((m) => m.role !== 'system')

	// in any case we want to prompt for the last user message
	const lastMessage = nonSystemMessages[nonSystemMessages.length - 1]
	if (lastMessage.role !== 'user') {
		throw new Error('Last message must be from user.')
	}
	const input = lastMessage.content

	// if context got reset, we need to reingest the chat history
	if (resetContext) {
		const historyItems: ChatHistoryItem[] = nonSystemMessages.map((m) => {
			return {
				type: m.role,
				text: m.content,
			} as ChatHistoryItem
		})
		instance.session.setChatHistory(historyItems)
	}

	let inputTokens = instance.model.tokenize(input)
	let generatedTokenCount = 0

	const stopTriggers: StopGenerationTrigger = []
	if (request.stop) {
		stopTriggers.push(...request.stop)
	}

	const result = await instance.session.promptWithMeta(input, {
		maxTokens: request.maxTokens,
		temperature: request.temperature,
		topP: request.topP,
		topK: request.topK,
		minP: request.minP,
		repeatPenalty: {
			lastTokens: request.repeatPenaltyNum ?? 64,
			frequencyPenalty: request.frequencyPenalty,
			presencePenalty: request.presencePenalty,
		},
		// TODO integrate stopGenerationTriggers
		// stopGenerationTriggers: stopTriggers.length ? [stopTriggers] : undefined,
		signal: signal,
		onToken: (tokens) => {
			generatedTokenCount++
			const text = instance.model.detokenize(tokens)
			if (onChunk) {
				onChunk({
					tokens,
					text,
				})
			}
		},
	})

	return {
		finishReason: result.stopReason,
		message: {
			role: 'assistant',
			content: result.responseText,
		},
		promptTokens: inputTokens.length,
		completionTokens: generatedTokenCount,
		totalTokens: inputTokens.length + generatedTokenCount,
	}
}

export async function processCompletion(
	instance: LlamaCppInstance,
	{ config, request, onChunk }: EngineCompletionContext<LlamaCppOptions>,
	signal?: AbortSignal,
): Promise<EngineCompletionResult> {
	if (!request.prompt) {
		throw new Error('Prompt is required for completion.')
	}

	if (instance.context) {
		await instance.context.dispose()
	}

	instance.context = await instance.model.createContext({
		createSignal: signal,
		seed: request.seed || createSeed(0, 1000000),
		threads: config.engineOptions?.cpuThreads,
		batchSize: config.engineOptions?.batchSize,
	})

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
