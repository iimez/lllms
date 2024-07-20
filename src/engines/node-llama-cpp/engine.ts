import path from 'node:path'
import fs from 'node:fs'
import { nanoid } from 'nanoid'
import {
	getLlama,
	LlamaOptions,
	LlamaChat,
	LlamaModel,
	LlamaContext,
	LlamaCompletion,
	LlamaLogLevel,
	LlamaChatResponseFunctionCall,
	TokenBias,
	Token,
	LlamaContextSequence,
	LlamaGrammar,
	ChatHistoryItem,
	LlamaChatResponse,
	ChatModelResponse,
	LlamaEmbeddingContext,
	defineChatSessionFunction,
	GbnfJsonSchema,
	ChatSessionModelFunction,
	createModelDownloader,
	readGgufFileInfo,
	GgufFileInfo,
	LlamaJsonSchemaGrammar,
	LlamaModelOptions,
	LLamaChatContextShiftOptions,
} from 'node-llama-cpp'
import { StopGenerationTrigger } from 'node-llama-cpp/dist/utils/StopGenerationDetector'
import {
	EngineChatCompletionResult,
	EngineTextCompletionResult,
	EngineTextCompletionArgs,
	EngineChatCompletionArgs,
	EngineContext,
	ToolDefinition,
	ToolCallResultMessage,
	AssistantMessage,
	EngineEmbeddingArgs,
	EngineEmbeddingResult,
	FileDownloadProgress,
	ModelConfig,
	TextCompletionParams,
	TextCompletionPreloadOptions,
	TextCompletionGrammar,
} from '#lllms/types/index.js'
import { LogLevels } from '#lllms/lib/logger.js'
import { flattenMessageTextContent } from '#lllms/lib/flattenMessageTextContent.js'
import { calculateFileChecksum } from '#lllms/lib/calculateFileChecksum.js'
import { acquireFileLock } from '#lllms/lib/acquireFileLock.js'
import {
	createSeed,
	createChatMessageArray,
	addFunctionCallToChatHistory,
	mapFinishReason,
} from './util.js'
import { LlamaChatResult } from './types.js'

export interface NodeLlamaCppInstance {
	model: LlamaModel
	context: LlamaContext
	chat?: LlamaChat
	chatHistory: ChatHistoryItem[]
	grammars: Record<string, LlamaGrammar>
	pendingFunctionCalls: Record<string, any>
	lastEvaluation?: LlamaChatResponse['lastEvaluation']
	embeddingContext?: LlamaEmbeddingContext
}

export interface NodeLlamaCppModelMeta {
	gguf: GgufFileInfo
}

export interface NodeLlamaCppModelConfig extends ModelConfig {
	location: string
	grammars?: Record<string, TextCompletionGrammar>
	sha256?: string
	completionDefaults?: TextCompletionParams
	tools?: Record<string, ToolDefinition>
	preload?: TextCompletionPreloadOptions
	contextSize?: number
	batchSize?: number
	lora?: LlamaModelOptions['lora']
	contextShiftStrategy?: LLamaChatContextShiftOptions['strategy']
	device?: {
		gpu?: boolean | 'auto' | (string & {})
		gpuLayers?: number
		cpuThreads?: number
		memLock?: boolean
	}
}

export const autoGpu = true

export async function prepareModel(
	{ config, log }: EngineContext<NodeLlamaCppModelConfig>,
	onProgress?: (progress: FileDownloadProgress) => void,
	signal?: AbortSignal,
) {
	fs.mkdirSync(path.dirname(config.location), { recursive: true })
	const clearFileLock = await acquireFileLock(config.location, signal)
	log(LogLevels.info, `Preparing node-llama-cpp model at ${config.location}`, {
		model: config.id,
	})
	if (!fs.existsSync(config.location)) {
		if (!config.url) {
			throw new Error(`Cannot download "${config.id}" - no URL configured`)
		}
		log(LogLevels.info, `Downloading ${config.id}`, {
			url: config.url,
			location: config.location,
		})
		const downloader = await createModelDownloader({
			modelUrl: config.url,
			dirPath: path.dirname(config.location),
			fileName: path.basename(config.location),
			deleteTempFileOnCancel: false,
			onProgress: (status) => {
				if (onProgress) {
					onProgress({
						file: config.location,
						loadedBytes: status.downloadedSize,
						totalBytes: status.totalSize,
					})
				}
			},
		})
		await downloader.download()
	}

	const postDownloadPromises: Array<Promise<GgufFileInfo | string>> = [
		readGgufFileInfo(config.location, {
			signal,
			ignoreKeys: [
				'gguf.tokenizer.ggml.merges',
				'gguf.tokenizer.ggml.tokens',
				'gguf.tokenizer.ggml.scores',
				'gguf.tokenizer.ggml.token_type',
			],
		}),
	]

	if (config.sha256) {
		postDownloadPromises.push(calculateFileChecksum(config.location, 'sha256'))
	}
	const [gguf, fileHash] = await Promise.all(postDownloadPromises)
	if (config.sha256 && fileHash !== config.sha256) {
		throw new Error(
			`Model sha256 checksum mismatch: expected ${config.sha256} got ${fileHash} for ${config.location}`,
		)
	}
	clearFileLock()
	return {
		gguf,
	}
}

export async function createInstance(
	{ config, log }: EngineContext<NodeLlamaCppModelConfig>,
	signal?: AbortSignal,
) {
	log(LogLevels.debug, 'Load Llama model', config.device)
	// takes "auto" | "metal" | "cuda" | "vulkan"
	const gpuSetting = (config.device?.gpu ?? 'auto') as LlamaOptions['gpu']
	const llama = await getLlama({
		gpu: gpuSetting,
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

	const llamaGrammars: Record<string, LlamaGrammar> = {
		json: await LlamaGrammar.getFor(llama, 'json'),
	}

	if (config.grammars) {
		for (const key in config.grammars) {
			const input = config.grammars[key]
			if (typeof input === 'string') {
				llamaGrammars[key] = new LlamaGrammar(llama, {
					grammar: input,
				})
			} else {
				// assume input is a JSON schema object
				llamaGrammars[key] = new LlamaJsonSchemaGrammar(
					llama,
					input as GbnfJsonSchema,
				)
			}
		}
	}

	const llamaModel = await llama.loadModel({
		modelPath: config.location, // full model absolute path
		loadSignal: signal,
		useMlock: config.device?.memLock ?? false,
		gpuLayers: config.device?.gpuLayers,
		lora: config.lora,
		// onLoadProgress: (percent) => {}
	})

	const context = await llamaModel.createContext({
		sequences: 1,
		seed: config.completionDefaults?.seed ?? createSeed(0, 1000000),
		threads: config.device?.cpuThreads,
		batchSize: config.batchSize,
		contextSize: config.contextSize,
		// batching: {
		// 	dispatchSchedule: 'nextTick',
		// 	itemPrioritizationStrategy: 'maximumParallelism',
		// 	itemPrioritizationStrategy: 'firstInFirstOut',
		// },
		createSignal: signal,
	})

	const instance: NodeLlamaCppInstance = {
		model: llamaModel,
		context,
		grammars: llamaGrammars,
		chat: undefined,
		chatHistory: [],
		pendingFunctionCalls: {},
		lastEvaluation: undefined,
	}

	if (config.preload) {
		// preloading chat session
		if ('messages' in config.preload) {
			const initialChatHistory = createChatMessageArray(config.preload.messages)
			const chat = new LlamaChat({
				contextSequence: context.getSequence(),
			})

			let inputFunctions: Record<string, ChatSessionModelFunction> | undefined
			if (config.tools && Object.keys(config.tools).length > 0) {
				inputFunctions = {}
				for (const functionName in config.tools) {
					const functionDef = config.tools[functionName]
					inputFunctions[functionName] = defineChatSessionFunction({
						description: functionDef.description,
						params: functionDef.parameters as GbnfJsonSchema,
						handler: functionDef.handler || (() => {}),
					}) as ChatSessionModelFunction
				}
			}

			const preloadRes = await chat.loadChatAndCompleteUserMessage(
				initialChatHistory,
				{
					initialUserPrompt: '',
					functions: inputFunctions,
					documentFunctionParams: config.preload.toolDocumentation,
				},
			)

			instance.chat = chat
			instance.chatHistory = initialChatHistory
			instance.lastEvaluation = {
				cleanHistory: initialChatHistory,
				contextWindow: preloadRes.lastEvaluation.contextWindow,
				contextShiftMetadata: preloadRes.lastEvaluation.contextShiftMetadata,
			}
		}

		if ('prefix' in config.preload) {
			// TODO preloading completion prefix
			// context.getSequence()
			// const completion = new LlamaCompletion({
			// 	contextSequence: context.getSequence(),
			// })
			// const tokens = model.tokenize(config.preload.prefix)
			// await completion.generateCompletion(tokens, {
			// 	maxTokens: 0,
			// })
			// completion.dispose()
		}
	}

	return instance
}

export async function disposeInstance(instance: NodeLlamaCppInstance) {
	await instance.model.dispose()
}

export async function processChatCompletionTask(
	{
		request,
		config,
		resetContext,
		log,
		onChunk,
	}: EngineChatCompletionArgs<NodeLlamaCppModelConfig>,
	instance: NodeLlamaCppInstance,
	signal?: AbortSignal,
): Promise<EngineChatCompletionResult> {
	if (!instance.chat || resetContext) {
		log(LogLevels.debug, 'Recreating chat context', {
			resetContext,
			willDisposeChat: !!instance.chat,
		})
		// if context reset is requested, dispose the chat instance
		if (instance.chat) {
			await instance.chat.dispose()
		}
		instance.chat = new LlamaChat({
			contextSequence: instance.context.getSequence(),
		})
		// reset state and reingest the conversation history
		instance.lastEvaluation = undefined
		instance.pendingFunctionCalls = {}
		instance.chatHistory = createChatMessageArray(request.messages)
		// drop last user message. its gonna be added later, after resolved function calls
		if (instance.chatHistory[instance.chatHistory.length - 1].type === 'user') {
			instance.chatHistory.pop()
		}
	}

	// set additional stop generation triggers for this completion
	const customStopTriggers: StopGenerationTrigger[] = []
	const stopTrigger = request.stop ?? config.completionDefaults?.stop
	if (stopTrigger) {
		customStopTriggers.push(...stopTrigger.map((t) => [t]))
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

	// setting up available function definitions
	const toolDefinitions: Record<string, ToolDefinition> = {
		...config.tools,
		...request.tools,
	}

	// see if the user submitted any function call results
	const resolvedFunctionCalls = []
	const functionCallResultMessages = request.messages.filter(
		(m) => m.role === 'tool',
	) as ToolCallResultMessage[]
	for (const message of functionCallResultMessages) {
		if (!instance.pendingFunctionCalls[message.callId]) {
			log(
				LogLevels.warn,
				`Received function result for non-existing call id "${message.callId}`,
			)
			continue
		}
		log(LogLevels.debug, 'Resolving pending function call', {
			id: message.callId,
			result: message.content,
		})
		const functionCall = instance.pendingFunctionCalls[message.callId]
		const functionDef = toolDefinitions[functionCall.functionName]
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
		delete instance.pendingFunctionCalls[message.callId]
	}
	// if we resolved any results, add them to history
	if (resolvedFunctionCalls.length) {
		instance.chatHistory.push({
			type: 'model',
			response: resolvedFunctionCalls.map((call) => {
				return {
					type: 'functionCall',
					...call,
				}
			}),
		})
	}

	// add the new user message to the chat history
	const lastMessage = request.messages[request.messages.length - 1]
	if (lastMessage.role === 'user' && lastMessage.content) {
		const newUserText = flattenMessageTextContent(lastMessage.content)
		if (newUserText) {
			instance.chatHistory.push({
				type: 'user',
				text: newUserText,
			})
		}
	} else if (!resolvedFunctionCalls.length) {
		throw new Error('Chat completions require a final user message.')
	}

	// only grammar or functions can be used, not both.
	// currently ignoring function definitions if grammar is provided

	let inputGrammar: LlamaGrammar | undefined
	let inputFunctions: Record<string, ChatSessionModelFunction> | undefined

	if (request.grammar) {
		if (!instance.grammars[request.grammar]) {
			throw new Error(`Grammar "${request.grammar}" not found.`)
		}
		inputGrammar = instance.grammars[request.grammar]
	} else if (Object.keys(toolDefinitions).length > 0) {
		inputFunctions = {}
		for (const functionName in toolDefinitions) {
			const functionDef = toolDefinitions[functionName]
			inputFunctions[functionName] = defineChatSessionFunction({
				description: functionDef.description,
				params: functionDef.parameters as GbnfJsonSchema,
				handler: functionDef.handler || (() => {}),
			}) as ChatSessionModelFunction
		}
	}
	const defaults = config.completionDefaults ?? {}
	let lastEvaluation: LlamaChatResponse['lastEvaluation'] | undefined =
		instance.lastEvaluation
	let newChatHistory = instance.chatHistory.slice()
	let newContextWindowChatHistory = !lastEvaluation?.contextWindow
		? undefined
		: instance.chatHistory.slice()

	if (instance.chatHistory[instance.chatHistory.length - 1].type !== 'model') {
		newChatHistory.push({
			type: 'model',
			response: [],
		})
		if (newContextWindowChatHistory) {
			newContextWindowChatHistory.push({
				type: 'model',
				response: [],
			})
		}
	}

	let completionResult: LlamaChatResult

	const inputTokenCountBefore =
		instance.chat.sequence.tokenMeter.usedInputTokens
	const outputTokenCountBefore =
		instance.chat.sequence.tokenMeter.usedOutputTokens

	const functionsOrGrammar = inputFunctions
		? {
				functions: inputFunctions,
				documentFunctionParams: true,
				maxParallelFunctionCalls: 2,
				onFunctionCall: (functionCall: LlamaChatResponseFunctionCall<any>) => {
					// log(LogLevels.debug, 'Called function', functionCall)
				},
		  }
		: {
				grammar: inputGrammar,
		  }

	while (true) {
		const {
			functionCalls,
			lastEvaluation: currentLastEvaluation,
			metadata,
		} = await instance.chat.generateResponse(newChatHistory, {
			signal,
			stopOnAbortSignal: true, // this will make aborted completions resolve (with a partial response)
			maxTokens: request.maxTokens ?? defaults.maxTokens,
			temperature: request.temperature ?? defaults.temperature,
			topP: request.topP ?? defaults.topP,
			topK: request.topK ?? defaults.topK,
			minP: request.minP ?? defaults.minP,
			tokenBias,
			customStopTriggers,
			trimWhitespaceSuffix: false,
			...functionsOrGrammar,
			repeatPenalty: {
				lastTokens: request.repeatPenaltyNum ?? defaults.repeatPenaltyNum,
				frequencyPenalty: request.frequencyPenalty ?? defaults.frequencyPenalty,
				presencePenalty: request.presencePenalty ?? defaults.presencePenalty,
			},
			contextShift: {
				strategy: config.contextShiftStrategy,
				lastEvaluationMetadata: lastEvaluation?.contextShiftMetadata,
			},
			lastEvaluationContextWindow: {
				history: newContextWindowChatHistory,
				minimumOverlapPercentageToPreventContextShift: 0.5,
			},
			onToken: (tokens) => {
				const text = instance.model.detokenize(tokens)
				if (onChunk) {
					onChunk({
						tokens,
						text,
					})
				}
			},
		})

		lastEvaluation = currentLastEvaluation
		newChatHistory = lastEvaluation.cleanHistory

		if (functionCalls) {
			// find leading immediately evokable function calls (=have a handler function)
			const evokableFunctionCalls = []
			for (const functionCall of functionCalls) {
				const functionDef = toolDefinitions[functionCall.functionName]
				if (functionDef.handler) {
					evokableFunctionCalls.push(functionCall)
				} else {
					break
				}
			}

			// resolve their results.
			const results = await Promise.all(
				evokableFunctionCalls.map(async (functionCall) => {
					const functionDef = toolDefinitions[functionCall.functionName]
					if (!functionDef) {
						throw new Error(
							`The model tried to call undefined function "${functionCall.functionName}"`,
						)
					}
					const functionCallResult = await functionDef.handler!(
						functionCall.params,
					)
					log(LogLevels.debug, 'Function handler resolved', {
						function: functionCall.functionName,
						args: functionCall.params,
						result: functionCallResult,
					})
					return {
						functionDef,
						functionCall,
						functionCallResult,
					}
				}),
			)
			newContextWindowChatHistory = lastEvaluation.contextWindow

			// add results to chat history in the order they were called
			for (const callResult of results) {
				newChatHistory = addFunctionCallToChatHistory({
					chatHistory: newChatHistory,
					functionName: callResult.functionCall.functionName,
					functionDescription: callResult.functionDef.description,
					callParams: callResult.functionCall.params,
					callResult: callResult.functionCallResult,
					rawCall: callResult.functionCall.raw,
				})
				newContextWindowChatHistory = addFunctionCallToChatHistory({
					chatHistory: newChatHistory,
					functionName: callResult.functionCall.functionName,
					functionDescription: callResult.functionDef.description,
					callParams: callResult.functionCall.params,
					callResult: callResult.functionCallResult,
					rawCall: callResult.functionCall.raw,
				})
			}

			// check if all function calls were immediately evokable
			const remainingFunctionCalls = functionCalls.slice(
				evokableFunctionCalls.length,
			)

			if (remainingFunctionCalls.length === 0) {
				// if yes, continue with generation
				lastEvaluation.cleanHistory = newChatHistory
				lastEvaluation.contextWindow = newContextWindowChatHistory!
				continue
			} else {
				// if no, return the function calls and skip generation
				completionResult = {
					responseText: null,
					stopReason: 'functionCalls',
					functionCalls: remainingFunctionCalls,
				}
				break
			}
		}

		// no function calls happened, we got a model response.
		instance.lastEvaluation = lastEvaluation
		instance.chatHistory = newChatHistory
		const lastMessage = instance.chatHistory[
			instance.chatHistory.length - 1
		] as ChatModelResponse
		const responseText = lastMessage.response
			.filter((item: any) => typeof item === 'string')
			.join('')
		completionResult = {
			responseText,
			stopReason: metadata.stopReason,
		}
		break
	}

	const assistantMessage: AssistantMessage = {
		role: 'assistant',
		content: completionResult.responseText || '',
	}

	if (completionResult.functionCalls) {
		// TODO its possible that there are tailing immediately-evaluatable function calls.
		// function call results need to be added in the order the functions were called, so
		// we need to wait for the pending calls to complete before we can add the tailing calls.
		// as is, these may never resolve
		const pendingFunctionCalls = completionResult.functionCalls.filter(
			(call) => {
				const functionDef = toolDefinitions[call.functionName]
				return !functionDef.handler
			},
		)

		// TODO write a test that triggers a parallel call to a deferred function and to an IE function
		const tailingFunctionCalls = completionResult.functionCalls.filter(
			(call) => {
				const functionDef = toolDefinitions[call.functionName]
				return functionDef.handler
			},
		)
		if (tailingFunctionCalls.length) {
			console.debug(tailingFunctionCalls)
			log(LogLevels.warn, 'Tailing function calls not resolved')
		}

		assistantMessage.toolCalls = pendingFunctionCalls.map((call) => {
			const callId = nanoid()
			instance.pendingFunctionCalls[callId] = call
			log(LogLevels.debug, 'Saving pending tool call', {
				id: callId,
				function: call.functionName,
				args: call.params,
			})
			return {
				id: callId,
				name: call.functionName,
				parameters: call.params,
			}
		})
	}

	const inputTokenCountAfter = instance.chat.sequence.tokenMeter.usedInputTokens
	const outputTokenCountAfter =
		instance.chat.sequence.tokenMeter.usedOutputTokens
	const promptTokens = inputTokenCountAfter - inputTokenCountBefore
	const completionTokens = outputTokenCountAfter - outputTokenCountBefore
	return {
		finishReason: mapFinishReason(completionResult.stopReason),
		message: assistantMessage,
		promptTokens,
		completionTokens,
		totalTokens: promptTokens + completionTokens,
	}
}

export async function processTextCompletionTask(
	{
		request,
		config,
		log,
		onChunk,
	}: EngineTextCompletionArgs<NodeLlamaCppModelConfig>,
	instance: NodeLlamaCppInstance,
	signal?: AbortSignal,
): Promise<EngineTextCompletionResult> {
	if (!request.prompt) {
		throw new Error('Prompt is required for text completion.')
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
			seed:
				request.seed ??
				config.completionDefaults?.seed ??
				createSeed(0, 1000000),
			threads: config.device?.cpuThreads,
			batchSize: config.batchSize,
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
		customStopTriggers: stopGenerationTriggers.length
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
		finishReason: mapFinishReason(result.metadata.stopReason),
		text: result.response,
		promptTokens: tokens.length,
		completionTokens: generatedTokenCount,
		totalTokens: tokens.length + generatedTokenCount,
	}
}

export async function processEmbeddingTask(
	{ request, config }: EngineEmbeddingArgs<NodeLlamaCppModelConfig>,
	instance: NodeLlamaCppInstance,
	signal?: AbortSignal,
): Promise<EngineEmbeddingResult> {
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

	if (!instance.embeddingContext) {
		instance.embeddingContext = await instance.model.createEmbeddingContext({
			batchSize: config.batchSize,
			createSignal: signal,
		})
	}

	const embeddings: Float32Array[] = []
	let inputTokens = 0

	for (const text of texts) {
		const tokenizedInput = instance.model.tokenize(text)
		inputTokens += tokenizedInput.length
		const embedding = await instance.embeddingContext.getEmbeddingFor(
			tokenizedInput,
		)
		embeddings.push(new Float32Array(embedding.vector))
		if (signal?.aborted) {
			break
		}
	}

	return {
		embeddings,
		inputTokens,
	}
}
