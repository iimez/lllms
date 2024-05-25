import {
	getLlama,
	LlamaChat,
	LlamaModel,
	LlamaContext,
	LlamaCompletion,
	LlamaLogLevel,
	TokenBias,
	Token,
	LlamaContextSequence,
	Llama,
	LlamaGrammar,
	ChatHistoryItem,
	LlamaChatResponse,
	ChatModelResponse,
	LlamaChatResponseFunctionCall,
	LlamaEmbeddingContext,
} from 'node-llama-cpp'
import {
	EngineChatCompletionResult,
	EngineCompletionResult,
	EngineCompletionContext,
	EngineChatCompletionContext,
	EngineContext,
	EngineOptionsBase,
	ChatCompletionFunction,
	FunctionCallResultMessage,
	AssistantMessage,
	EngineEmbeddingContext,
	EngineEmbeddingResult,
} from '#lllms/types/index.js'
import { LogLevels } from '#lllms/lib/logger.js'
import { formatBytes, mergeAbortSignals } from '#lllms/lib/util.js'
import { nanoid } from 'nanoid'
import { StopGenerationTrigger } from 'node-llama-cpp/dist/utils/StopGenerationDetector'

// https://github.com/withcatai/node-llama-cpp/pull/105
// https://github.com/withcatai/node-llama-cpp/discussions/109

export interface LlamaCppOptions extends EngineOptionsBase {
	memLock?: boolean
}

interface LlamaCppInstance {
	model: LlamaModel
	context: LlamaContext
	chat: LlamaChat
	messages: ChatHistoryItem[]
	grammars: Record<string, LlamaGrammar>
	pendingFunctionCalls: Record<string, any>
	lastEvaluation?: LlamaChatResponse['lastEvaluation']
	embeddingContext?: LlamaEmbeddingContext
}

interface LlamaChatFunction {
	description?: string
	params?: any
	handler?: (params: any) => any
}

interface LlamaChatResult {
	responseText: string | null
	functionCalls?: LlamaChatResponseFunctionCall<any>[]
	stopReason: LlamaChatResponse['metadata']['stopReason']
}

function prepareGrammars(llama: Llama, grammarConfig: Record<string, string>) {
	const grammars: Record<string, LlamaGrammar> = {}
	for (const key in grammarConfig) {
		const grammar = new LlamaGrammar(llama, {
			grammar: grammarConfig[key],
			// printGrammar: true,
		})
		grammars[key] = grammar
	}
	return grammars
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

	let grammars: Record<string, LlamaGrammar> = {}
	if (config.grammars) {
		grammars = prepareGrammars(llama, config.grammars)
	}

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
		grammars,
		chat: null,
		pendingFunctionCalls: {},
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

function addFunctionCallToChatHistory({
	chatHistory,
	functionName,
	functionDescription,
	callParams,
	callResult,
	raw,
}: any): ChatHistoryItem[] {
	const newChatHistory = chatHistory.slice()
	if (
		newChatHistory.length === 0 ||
		newChatHistory[newChatHistory.length - 1].type !== 'model'
	)
		newChatHistory.push({
			type: 'model',
			response: [],
		})
	const lastModelResponseItem = newChatHistory[newChatHistory.length - 1]
	const newLastModelResponseItem = { ...lastModelResponseItem }
	newChatHistory[newChatHistory.length - 1] = newLastModelResponseItem
	const modelResponse = newLastModelResponseItem.response.slice()
	newLastModelResponseItem.response = modelResponse
	modelResponse.push({
		type: 'functionCall',
		name: functionName,
		description: functionDescription,
		params: callParams,
		result: callResult,
		raw,
	})
	return newChatHistory
}

export async function processChatCompletion(
	instance: LlamaCppInstance,
	{
		request,
		config,
		resetContext,
		log,
		onChunk,
	}: EngineChatCompletionContext<LlamaCppOptions>,
	signal?: AbortSignal,
): Promise<EngineChatCompletionResult> {
	const conversationMessages = request.messages.filter(
		(m) => m.role !== 'system',
	)

	if (!instance.chat || !instance.chat.context.sequencesLeft || resetContext) {
		if (instance.chat && !instance.chat?.context?.sequencesLeft) {
			log(LogLevels.debug, 'No sequencesLeft, recreating LlamaChat')
		}
		if (!instance.chat) {
			log(LogLevels.debug, 'Creating LlamaChat')
		}
		if (instance.chat) {
			await instance.chat.dispose()
		}
		let systemPrompt = request.systemPrompt ?? config.systemPrompt
		if (!systemPrompt && request.messages[0].role === 'system') {
			systemPrompt = request.messages[0].content
		}
		instance.messages = []
		if (systemPrompt) {
			instance.messages.push({
				type: 'system',
				text: systemPrompt,
			})
		}

		instance.chat = new LlamaChat({
			contextSequence: instance.context.getSequence(),
			// contextShift: {
			// 	size: 50,
			// 	strategy: 'eraseFirstResponseAndKeepFirstSystem',
			// },
		})
	}

	// if context has been reset we need to reingest the conversation history
	if (resetContext) {
		const newMessages: ChatHistoryItem[] = []
		for (const message of conversationMessages) {
			newMessages.push({
				type: message.role,
				text: message.content,
			} as ChatHistoryItem)
		}
		// drop last user message, thats what we wanna prompt with.
		if (newMessages[newMessages.length - 1].type === 'user') {
			newMessages.pop()
		}
		instance.messages.push(...newMessages)
	}

	// set additional stop generation triggers for this completion
	const stopGenerationTriggers: StopGenerationTrigger[] = []
	const stopTrigger = request.stop ?? config.completionDefaults?.stop
	if (stopTrigger) {
		stopGenerationTriggers.push(...stopTrigger.map((t) => [t]))
	}
	// setting up logit/token bias dictionary
	let tokenBias: TokenBias | undefined
	const completionTokenBias =
		request.tokenBias ?? config.completionDefaults?.tokenBias
	if (completionTokenBias) {
		tokenBias = new TokenBias(instance.model)
		for (const key in completionTokenBias) {
			const bias = completionTokenBias[key] / 10
			const tokenId = parseInt(key) as Token
			if (!isNaN(tokenId)) {
				tokenBias.set(tokenId, bias)
			} else {
				tokenBias.set(key, bias)
			}
		}
	}

	let inputTokenCount = 0
	let generatedTokenCount = 0
	let completionResult: EngineChatCompletionResult

	// setting up available function definitions
	const functionDefinitions: Record<string, ChatCompletionFunction> = {
		...config.functions,
		...request.functions,
	}
	
	// see if the user submitted any function call results
	const resolvedFunctionCalls = []
	const functionCallResultMessages = request.messages.filter(
		(m) => m.role === 'function',
	) as FunctionCallResultMessage[]
	for (const message of functionCallResultMessages) {
		if (instance.pendingFunctionCalls[message.callId]) {
			log(LogLevels.debug, 'Resolving pending function call', message)
			const functionCall = instance.pendingFunctionCalls[message.callId]
			const functionDef = functionDefinitions[functionCall.functionName]
			resolvedFunctionCalls.push({
				name: functionCall.functionName,
				description: functionDef?.description,
				params: functionCall.params,
				result: message.content,
				raw:
					functionCall.raw +
					instance.chat.chatWrapper.generateFunctionCallResult(
						functionCall.functionName,
						functionCall.params,
						message.content,
					),
			})
		} else {
			log(LogLevels.warn, 'Pending function call not found', message)
		}
	}
	if (resolvedFunctionCalls.length) {
		instance.messages.push({
			type: 'model',
			response: resolvedFunctionCalls.map((call) => {
				return {
					type: 'functionCall',
					...call,
				}
			}),
		})
	}

	let newUserMessage: string | undefined
	const lastMessage = conversationMessages[conversationMessages.length - 1]
	if (lastMessage.role === 'user') {
		newUserMessage = lastMessage.content
	}

	if (newUserMessage) {
		inputTokenCount = instance.model.tokenize(newUserMessage).length
		instance.messages.push({
			type: 'user',
			text: newUserMessage,
		})
	}
	
	// only grammar or functions can be used, not both.
	// currently ignoring function definitions if grammar is provided

	let inputGrammar: LlamaGrammar | undefined
	let inputFunctions: Record<string, LlamaChatFunction> | undefined

	if (request.grammar) {
		if (!instance.grammars[request.grammar]) {
			throw new Error(`Grammar "${request.grammar}" not found.`)
		}
		inputGrammar = instance.grammars[request.grammar]
	} else if (Object.keys(functionDefinitions).length > 0) {
		inputFunctions = {}
		for (const functionName in functionDefinitions) {
			const functionDef = functionDefinitions[functionName]
			inputFunctions[functionName] = {
				description: functionDef.description,
				params: functionDef.parameters,
				handler: functionDef.handler,
			}
		}
	}

	const defaults = config.completionDefaults ?? {}

	let lastEvaluation: LlamaChatResponse['lastEvaluation'] | undefined
	let newChatHistory = instance.messages.slice()
	let newContextWindowChatHistory =
		lastEvaluation?.contextWindow == null
			? undefined
			: instance.messages.slice()

	newChatHistory.push({
		type: 'model',
		response: [],
	})
	if (newContextWindowChatHistory != null) {
		newContextWindowChatHistory.push({
			type: 'model',
			response: [],
		})
	}

	let result: LlamaChatResult

	while (true) {
		const {
			functionCall,
			lastEvaluation: currentLastEvaluation,
			metadata,
		} = await instance.chat.generateResponse(newChatHistory, {
			maxTokens: request.maxTokens ?? defaults.maxTokens,
			temperature: request.temperature ?? defaults.temperature,
			topP: request.topP ?? defaults.topP,
			topK: request.topK ?? defaults.topK,
			minP: request.minP ?? defaults.minP,
			tokenBias,
			grammar: inputGrammar,
			// @ts-ignore we have asserted that grammar and functions are mutually exclusive
			functions: inputFunctions,
			// @ts-ignore we have asserted that grammar and functions are mutually exclusive
			documentFunctionParams: !!inputFunctions,
			customStopTriggers: stopGenerationTriggers,
			repeatPenalty: {
				lastTokens: request.repeatPenaltyNum ?? defaults.repeatPenaltyNum,
				frequencyPenalty: request.frequencyPenalty ?? defaults.frequencyPenalty,
				presencePenalty: request.presencePenalty ?? defaults.presencePenalty,
			},
			signal,
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
			contextShift: {
				lastEvaluationMetadata: lastEvaluation?.contextShiftMetadata,
			},
			lastEvaluationContextWindow: {
				history: newContextWindowChatHistory,
				minimumOverlapPercentageToPreventContextShift: 0.01,
			},
		})

		lastEvaluation = currentLastEvaluation
		newChatHistory = lastEvaluation.cleanHistory
		if (functionCall) {
			const functionDef = functionDefinitions[functionCall.functionName]
			if (functionDef == null) {
				throw new Error(
					`The model tried to call function "${functionCall.functionName}" which is not defined`,
				)
			}
			console.debug('function called', {
				functionCall,
				currentLastEvaluation,
				metadata,
			})
			if (functionDef.handler) {
				const functionCallResult = await functionDef.handler(
					functionCall.params,
				)
				log(LogLevels.debug, 'Called function handler', {
					name: functionCall.functionName,
					params: functionCall.params,
					result: functionCallResult,
				})
				newChatHistory = addFunctionCallToChatHistory({
					chatHistory: newChatHistory,
					functionName: functionCall.functionName,
					functionDescription: functionDef.description,
					callParams: functionCall.params,
					callResult: functionCallResult,
					raw:
						functionCall.raw +
						instance.chat.chatWrapper.generateFunctionCallResult(
							functionCall.functionName,
							functionCall.params,
							functionCallResult,
						),
				})
				newContextWindowChatHistory = addFunctionCallToChatHistory({
					chatHistory: lastEvaluation.contextWindow,
					functionName: functionCall.functionName,
					functionDescription: functionDef.description,
					callParams: functionCall.params,
					callResult: functionCallResult,
					raw:
						functionCall.raw +
						instance.chat.chatWrapper.generateFunctionCallResult(
							functionCall.functionName,
							functionCall.params,
							functionCallResult,
						),
				})
				lastEvaluation.cleanHistory = newChatHistory
				lastEvaluation.contextWindow = newContextWindowChatHistory
				continue
			} else {
				log(LogLevels.debug, 'Function without handler called', functionCall)
				result = {
					responseText: null,
					stopReason: 'functionCall',
					functionCalls: [functionCall],
				}
				break
			}
		}
		instance.messages = newChatHistory
		const lastMessage = instance.messages[
			instance.messages.length - 1
		] as ChatModelResponse
		// console.debug('lastMessage', JSON.stringify(lastMessage, null, 2))
		const responseText = lastMessage.response
			.filter((item: any) => typeof item === 'string')
			.join('')
		let stopReason = metadata.stopReason
		if (stopReason === 'customStopTrigger') {
			stopReason = 'stopGenerationTrigger'
		}
		result = {
			responseText,
			stopReason,
		}
		break
	}

	const assistantMessage: AssistantMessage = {
		role: 'assistant',
		content: result.responseText,
	}

	if (result.functionCalls?.length) {
		assistantMessage.functionCalls = result.functionCalls.map((call) => {
			const callId = nanoid()
			instance.pendingFunctionCalls[callId] = call
			log(LogLevels.debug, 'Adding pending function call result', call)
			return {
				id: callId,
				name: call.functionName,
				parameters: call.params,
			}
		})
	}

	completionResult = {
		finishReason: result.stopReason,
		message: assistantMessage,
		promptTokens: inputTokenCount,
		completionTokens: generatedTokenCount,
		totalTokens: inputTokenCount + generatedTokenCount,
	}
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

	let contextSequence: LlamaContextSequence
	if (instance.context.sequencesLeft) {
		log(LogLevels.debug, 'Clearing history', {
			sequencesLeft: instance.context.sequencesLeft,
		})
		contextSequence = instance.context.getSequence()
		await contextSequence.clearHistory()
	} else {
		log(LogLevels.debug, 'No sequencesLeft, recreating context')
		await instance.context.dispose()
		instance.context = await instance.model.createContext({
			createSignal: signal,
			seed: request.seed ?? config.completionDefaults?.seed, // || createSeed(0, 1000000),
			threads: config.engineOptions?.cpuThreads,
			batchSize: config.engineOptions?.batchSize,
		})
		contextSequence = instance.context.getSequence()
	}

	const completion = new LlamaCompletion({
		contextSequence: contextSequence,
	})

	const stopGenerationTriggers: StopGenerationTrigger[] = []
	const stopTrigger = request.stop ?? config.completionDefaults?.stop
	if (stopTrigger) {
		stopGenerationTriggers.push(...stopTrigger.map((t) => [t]))
	}

	const tokens = instance.model.tokenize(request.prompt)
	const defaults = config.completionDefaults ?? {}
	let generatedTokenCount = 0
	const result = await completion.generateCompletionWithMeta(tokens, {
		maxTokens: request.maxTokens ?? defaults.maxTokens,
		temperature: request.temperature ?? defaults.temperature,
		topP: request.topP ?? defaults.topP,
		topK: request.topK ?? defaults.topK,
		minP: request.minP ?? defaults.minP,
		repeatPenalty: {
			lastTokens: request.repeatPenaltyNum ?? defaults.repeatPenaltyNum,
			frequencyPenalty: request.frequencyPenalty ?? defaults.frequencyPenalty,
			presencePenalty: request.presencePenalty ?? defaults.presencePenalty,
		},
		signal: signal,
		stopGenerationTriggers: stopGenerationTriggers.length
			? stopGenerationTriggers
			: undefined,
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


export async function processEmbedding(
	instance: LlamaCppInstance,
	{
		request,
		config,
	}: EngineEmbeddingContext<LlamaCppOptions>,
	signal?: AbortSignal,
): Promise<EngineEmbeddingResult> {
	
	const texts: string[] = []
	if (typeof request.input === 'string') {
		texts.push(request.input)
	} else {
		const strInputs = request.input.filter((i) => typeof i === 'string') as string[]
		texts.push(...strInputs)
	}
	
	if (!instance.embeddingContext) {
		// console.debug('creating embed context')
		instance.embeddingContext = await instance.model.createEmbeddingContext();
	}

	const embeddings: Float32Array[] = []
	let inputTokens = 0
	
	for (const text of texts) {
		const tokenizedInput = instance.model.tokenize(text)
		inputTokens += tokenizedInput.length
		const embedding = await instance.embeddingContext.getEmbeddingFor(tokenizedInput);
		embeddings.push(new Float32Array(embedding.vector))
	}
	
	return {
		embeddings,
		inputTokens,
	}
	
}