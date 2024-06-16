import type {
	EngineType,
	EngineInstance,
	LlamaCppOptions,
	GPT4AllOptions,
} from '#lllms/engines/index.js'
import type { Logger } from '#lllms/lib/logger.js'
import type { SchemaObject, JSONSchemaType } from 'ajv'
import type { GbnfJsonSchema, GbnfJsonObjectSchema } from 'node-llama-cpp'

export type LLMTaskType = 'inference' | 'embedding'

export type CompletionFinishReason =
	| 'maxTokens'
	| 'functionCall'
	| 'eogToken'
	| 'stopTrigger'
	| 'abort'
	| 'cancel'
	| 'timeout'

export interface CompletionChunk {
	tokens: number[]
	text: string
}

export interface CompletionProcessingOptions {
	timeout?: number
	signal?: AbortSignal
	onChunk?: (chunk: CompletionChunk) => void
}

export interface AssistantFunctionCall {
	id: string
	name: string
	parameters?: Record<string, any>
}

export type ChatMessage = UserMessage | SystemMessage | AssistantMessage | FunctionCallResultMessage

export interface UserMessage {
	role: 'user'
	content: string
}

export interface SystemMessage {
	role: 'system'
	content: string
}

export interface AssistantMessage {
	role: 'assistant'
	content: string
	functionCalls?: AssistantFunctionCall[]
}

export interface FunctionCallResultMessage {
	role: 'function'
	content: string
	callId: string
	name: string
}

export interface CompletionParams {
	temperature?: number
	maxTokens?: number
	seed?: number
	stop?: string[]
	repeatPenalty?: number
	repeatPenaltyNum?: number
	frequencyPenalty?: number
	presencePenalty?: number
	grammar?: string
	topP?: number
	minP?: number
	topK?: number
	tokenBias?: Record<string, number>
}

export interface CompletionRequestBase extends CompletionParams {
	model: string
	stream?: boolean
}

export interface CompletionRequest extends CompletionRequestBase {
	prompt?: string
}

// TODO figure out how to type this better.
// export type FunctionDefinitionParams<TParamList extends string = any> = GbnfJsonObjectSchema
// export type FunctionDefinitionParams<TParams = any> = JSONSchemaType<TParams>
export type FunctionDefinitionParams<TParams = any> = Record<string, unknown>;

export interface FunctionDefinition<TParams = any> {
	description?: string
	parameters?: FunctionDefinitionParams<TParams>
	handler?: (params: TParams) => Promise<string>
}

export interface ChatCompletionRequest extends CompletionRequestBase {
	systemPrompt?: string
	messages: ChatMessage[]
	grammar?: string
	functions?: Record<string, FunctionDefinition>
}

export interface EmbeddingsRequest {
	model: string
	input: string | string[] | number[] | number[][];
	dimensions?: number
}

export type IncomingLLMRequest = CompletionRequest | ChatCompletionRequest | EmbeddingsRequest
export interface LLMRequestMeta {
	sequence: number
}
export type LLMRequest = LLMRequestMeta & IncomingLLMRequest

export interface ChatCompletionResult extends EngineChatCompletionResult {
	id: string
	model: string
}

export interface LLMOptionsBase {
	url?: string
	file?: string
	engine: EngineType
	task: LLMTaskType
	prepare?: 'blocking' | 'async' | 'on-demand'
	contextSize?: number
	minInstances?: number
	maxInstances?: number
	systemPrompt?: string
	grammars?: Record<string, string>
	functions?: Record<string, FunctionDefinition>
	completionDefaults?: CompletionParams
	md5?: string
	sha256?: string
}

export interface LLMConfig<T extends EngineOptionsBase = EngineOptionsBase>
	extends LLMOptionsBase {
	id: string
	file: string
	ttl?: number
	task: LLMTaskType
	engine: EngineType
	engineOptions?: T
}

export interface LLMEngine<T extends EngineOptionsBase = EngineOptionsBase> {
	loadInstance: (
		ctx: EngineContext<T>,
		signal?: AbortSignal,
	) => Promise<EngineInstance>
	disposeInstance: (instance: EngineInstance) => Promise<void>
	processChatCompletion: (
		instance: EngineInstance,
		ctx: EngineChatCompletionContext<T>,
		signal?: AbortSignal,
	) => Promise<EngineChatCompletionResult>
	processCompletion: (
		instance: EngineInstance,
		ctx: EngineCompletionContext<T>,
		signal?: AbortSignal,
	) => Promise<EngineCompletionResult>
	processEmbeddings: (
		instance: EngineInstance,
		ctx: EngineEmbeddingContext<T>,
		signal?: AbortSignal,
	) => Promise<EngineEmbeddingsResult>
}

export interface EngineCompletionContext<T extends EngineOptionsBase>
	extends EngineContext<T> {
	onChunk?: (chunk: CompletionChunk) => void
	request: CompletionRequest
}

export interface EngineChatCompletionContext<T extends EngineOptionsBase>
	extends EngineContext<T> {
	onChunk?: (chunk: CompletionChunk) => void
	resetContext?: boolean
	request: ChatCompletionRequest
}

export interface EngineEmbeddingContext<T extends EngineOptionsBase>
	extends EngineContext<T> {
	request: EmbeddingsRequest
}

export interface EngineEmbeddingsResult {
	embeddings: Float32Array[]
	inputTokens: number
}

export interface EngineContext<
	T extends EngineOptionsBase = EngineOptionsBase,
> {
	config: LLMConfig<T>
	log: Logger
}

export interface EngineChatCompletionResult {
	message: AssistantMessage
	finishReason: CompletionFinishReason
	promptTokens: number
	completionTokens: number
	totalTokens: number
}

export interface EngineCompletionResult {
	text: string
	finishReason?: CompletionFinishReason
	promptTokens: number
	completionTokens: number
	totalTokens: number
}

export interface EngineOptionsBase {
	gpu?: boolean | 'auto' | string
	gpuLayers?: number
	batchSize?: number
	cpuThreads?: number
}

interface NodeLlamaCppLLMOptions extends LLMOptionsBase {
	engine: 'node-llama-cpp'
	engineOptions?: LlamaCppOptions
}

interface GPT4AllLLMOptions extends LLMOptionsBase {
	engine: 'gpt4all'
	engineOptions?: GPT4AllOptions
}

// interface DefaultEngineLLMOptions extends LLMOptionsBase {
// 	engineOptions?: LlamaCppOptions
// }

export type LLMOptions =
	| NodeLlamaCppLLMOptions
	| GPT4AllLLMOptions
	// | DefaultEngineLLMOptions
