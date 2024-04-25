import {
	getLlama,
	LlamaChatSession,
	LlamaModel,
	LlamaContext,
	ChatMLChatWrapper,
	LlamaCompletion,
	Llama3ChatWrapper,
	Token,
} from 'node-llama-cpp'
import {
	LLMConfig,
	CompletionRequest,
	EngineChatCompletionResult,
	EngineCompletionResult,
	GenerationArgs,
} from '../types/index.js'
import { StopGenerationTrigger } from 'node-llama-cpp/dist/utils/StopGenerationDetector.js'

interface LlamaCppInstance {
	model: LlamaModel
	context: LlamaContext
	session: LlamaChatSession | null
}

export async function loadInstance(config: LLMConfig, signal?: AbortSignal) {
	console.debug('Creating llama instance', config)

	// https://github.com/withcatai/node-llama-cpp/pull/105
	// https://github.com/withcatai/node-llama-cpp/discussions/109

	const llama = await getLlama({
		gpu: config.gpu ? 'auto' : false, // "auto" | "metal" | "cuda" | "vulkan"
		// logLevel: 'warn',
		// logger: (level, message) => {},
	})
	const model = await llama.loadModel({
		modelPath: config.file, // full model absolute path
		// useMlock: false,
		loadSignal: signal,
	})

	const context = await model.createContext({
		// sequences: 1,
		// contextSize: "auto",
		// seed: 0,
		// threads: 4, // 0 = max
		// sequences: 1,
		// batchSize: 128,
		// contextSize: 2048,
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

function pickTemplateFormatter(templateFormat: string | undefined) {
	switch (templateFormat) {
		case 'chatml':
			return new ChatMLChatWrapper()
		case 'llama3':
			return new Llama3ChatWrapper()
		default:
			return new ChatMLChatWrapper()
	}
}

export async function processChatCompletion(
	instance: LlamaCppInstance,
	request: CompletionRequest,
	args: GenerationArgs,
): Promise<EngineChatCompletionResult> {
	if (!request.messages) {
		throw new Error('Messages are required for chat completion.')
	}
	if (!instance.session) {
		let systemPrompt = request.systemPrompt
		if (!systemPrompt && request.messages[0].role === 'system') {
			systemPrompt = request.messages[0].content
		}
		// if (systemPrompt) {
		// 	console.debug('using system prompt', systemPrompt)
		// }
		instance.session = new LlamaChatSession({
			contextSequence: instance.context.getSequence(),
			chatWrapper: pickTemplateFormatter(request.templateFormat),
			systemPrompt,
		})
	}

	const nonSystemMessages = request.messages.filter((m) => m.role !== 'system')

	const input = nonSystemMessages.map((m) => m.content).join('\n')
	let inputTokens = instance.model.tokenize(input)
	let generatedTokenCount = 0

	const result = await instance.session.promptWithMeta(input, {
		maxTokens: request.maxTokens,
		temperature: request.temperature,
		topP: request.topP,
		repeatPenalty: {
			frequencyPenalty: request.frequencyPenalty,
			presencePenalty: request.presencePenalty,
		},
		// stop // TODO integrate stopGenerationTriggers
		// topK: completionArgs.topK,
		// minP: completionArgs.minP,
		signal: args.signal,
		onToken: (tokens) => {
			generatedTokenCount++
			const text = instance.model.detokenize(tokens) // TODO will this break emojis?
			// console.debug('onToken', {tokens, text})
			if (args.onChunk) {
				args.onChunk({
					tokenId: tokens[0],
					token: text,
				})
			}
		},
	})

	return {
		finishReason: result.stopReason,
		message: {
			content: result.responseText,
			role: 'assistant',
		},
		promptTokens: inputTokens.length,
		completionTokens: generatedTokenCount,
		totalTokens: inputTokens.length + generatedTokenCount,
	}
}

export async function processCompletion(
	instance: LlamaCppInstance,
	request: CompletionRequest,
	args: GenerationArgs,
): Promise<EngineCompletionResult> {
	if (!request.prompt) {
		throw new Error('Prompt is required for completion.')
	}
	
	// console.debug('context size', instance.context.getAllocatedContextSize())
	
	// instance.context.getAllocatedContextSize()
	instance.context = await instance.model.createContext({
		createSignal: args.signal,
	})

	// console.debug('context', {
	// 	allocated: instance.context.getAllocatedContextSize(),
	// 	sequencesLeft: instance.context.sequencesLeft,
	// })

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
		signal: args.signal,
		stopGenerationTriggers: stopTriggers.length ? [stopTriggers] : undefined,
		onToken: (tokens) => {
			generatedTokenCount++
			const text = instance.model.detokenize(tokens)
			if (args.onChunk) {
				args.onChunk({
					tokenId: tokens[0],
					token: text,
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
