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
} from 'node-llama-cpp'
import { StopGenerationTrigger } from 'node-llama-cpp/dist/utils/StopGenerationDetector.js'
import {
	LLMConfig,
	CompletionRequest,
	ChatCompletionRequest,
	EngineChatCompletionResult,
	EngineCompletionResult,
	EngineCompletionContext,
	EngineContext,
} from '../types/index.js'
import { LogLevels } from '../util/log.js'

interface LlamaCppInstance {
	model: LlamaModel
	context: LlamaContext
	session: LlamaChatSession | null
}

// https://github.com/withcatai/node-llama-cpp/pull/105
// https://github.com/withcatai/node-llama-cpp/discussions/109

export async function loadInstance(config: LLMConfig, ctx: EngineContext) {
	ctx.logger(LogLevels.verbose, `Load Llama model from ${config.file}`, {
		instance: ctx.instance,
	})


	const llama = await getLlama({
		gpu: config.gpu ? 'auto' : false, // "auto" | "metal" | "cuda" | "vulkan"
		// logLevel: 'warn',
		logLevel: LlamaLogLevel.info,
		logger: (level, message) => {
			if (level === LlamaLogLevel.warn) {
				ctx.logger(LogLevels.warn, message, { instance: ctx.instance })
			} else if (
				level === LlamaLogLevel.error ||
				level === LlamaLogLevel.fatal
			) {
				ctx.logger(LogLevels.error, message, { instance: ctx.instance })
			} else if (level === LlamaLogLevel.info) {
				ctx.logger(LogLevels.verbose, message, { instance: ctx.instance })
			}
		},
	})
	const model = await llama.loadModel({
		modelPath: config.file, // full model absolute path
		loadSignal: ctx.signal,
		// useMlock: false,
		// onLoadProgress: (percent) => {}
	})

	const context = await model.createContext({
		sequences: 2,
		// TODO
		// seed: ,
		// threads: 4, // 0 = max
		// batchSize: 128,
		createSignal: ctx.signal,
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

function pickTemplateFormatter(
	templateFormat: string | undefined,
): ChatWrapper | undefined {
	switch (templateFormat) {
		case 'chatml':
			return new ChatMLChatWrapper()
		case 'llama3':
			return new Llama3ChatWrapper()
		case 'phi':
			return new GeneralChatWrapper({
				userMessageTitle: '<|user|>',
				modelResponseTitle: '<|assistant|>',
			})
	}
	return undefined
}

export async function processChatCompletion(
	instance: LlamaCppInstance,
	request: ChatCompletionRequest,
	ctx: EngineCompletionContext,
): Promise<EngineChatCompletionResult> {
	if (
		ctx.resetContext ||
		!instance.session ||
		!instance.context.sequencesLeft
	) {
		// allow setting system prompt via initial message.
		let systemPrompt = request.systemPrompt
		if (!systemPrompt && request.messages[0].role === 'system') {
			systemPrompt = request.messages[0].content
		}
		if (!instance.context.sequencesLeft) {
			ctx.logger(LogLevels.warn, 'No sequences left, recreating context.')
			instance.context.dispose()
			instance.context = await instance.model.createContext({
				createSignal: ctx.signal,
				sequences: 2,
				contextSize: 4096,
			})
		}
		if (instance.session) {
			instance.session.dispose()
		}
		instance.session = new LlamaChatSession({
			chatWrapper: pickTemplateFormatter(request.templateFormat),
			systemPrompt,
			contextSequence: instance.context.getSequence(),
			autoDisposeSequence: true,
			// contextShift: {
			// 	size: 50,
			// 	strategy: "eraseFirstResponseAndKeepFirstSystem"
			// },
		})
	}

	const nonSystemMessages = request.messages.filter((m) => m.role !== 'system')

	// in any case we want to prompt for the last user message
	const lastMessage = nonSystemMessages[nonSystemMessages.length - 1]
	if (lastMessage.role !== 'user') {
		throw new Error('Last message must be from user.')
	}
	const input = lastMessage.content

	// if context got reset, we need to reingest the chat history
	if (ctx.resetContext) {
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
			frequencyPenalty: request.frequencyPenalty,
			presencePenalty: request.presencePenalty,
		},
		// TODO integrate stopGenerationTriggers
		// stopGenerationTriggers: stopTriggers.length ? [stopTriggers] : undefined,
		signal: ctx.signal,
		onToken: (tokens) => {
			generatedTokenCount++
			const text = instance.model.detokenize(tokens)
			if (ctx.onChunk) {
				ctx.onChunk({
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
	request: CompletionRequest,
	ctx: EngineCompletionContext,
): Promise<EngineCompletionResult> {
	if (!request.prompt) {
		throw new Error('Prompt is required for completion.')
	}

	instance.context = await instance.model.createContext({
		createSignal: ctx.signal,
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
		repeatPenalty: {
			frequencyPenalty: request.frequencyPenalty,
			presencePenalty: request.presencePenalty,
		},
		signal: ctx.signal,
		stopGenerationTriggers: stopTriggers.length ? [stopTriggers] : undefined,
		onToken: (tokens) => {
			generatedTokenCount++
			const text = instance.model.detokenize(tokens)
			if (ctx.onChunk) {
				ctx.onChunk({
					tokens,
					text,
				})
			}
		},
	})

	return {
		finishReason: result.metadata.stopReason,
		text: result.response,
		promptTokens: tokens.length,
		completionTokens: generatedTokenCount,
		totalTokens: tokens.length + generatedTokenCount,
	}
}
