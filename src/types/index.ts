import type {
	BuiltInEngineName,
	NodeLlamaCppEngineOptions,
	GPT4AllEngineOptions,
	TransformersJsEngineOptions,
} from '#lllms/engines/index.js'
import type { Logger } from '#lllms/lib/logger.js'
import type { ModelPool } from '#lllms/pool.js'
import { ModelStore } from '#lllms/store.js'
import {
	AssistantMessage,
	ChatMessage,
	CompletionFinishReason,
	CompletionChunk,
	TextCompletionParams,
	ToolDefinition,
	TextCompletionPreloadOptions,
} from '#lllms/types/completions.js'
export * from '#lllms/types/completions.js'

export type ModelTaskType = 'text-completion' | 'embedding' | 'image-to-text'

export interface TextCompletionRequestBase extends TextCompletionParams {
	model: string
	stream?: boolean
}

export interface TextCompletionRequest extends TextCompletionRequestBase {
	prompt?: string
}

export interface ChatCompletionRequest extends TextCompletionRequestBase {
	messages: ChatMessage[]
	grammar?: string
	tools?: Record<string, ToolDefinition>
}

export interface EmbeddingRequest {
	model: string
	input: string | string[] | number[] | number[][]
	dimensions?: number
}

export interface ImageToTextRequest {
	model: string
	url?: string
	file?: string
}

export interface ModelRequestMeta {
	sequence: number
}
export type IncomingRequest =
	| TextCompletionRequest
	| ChatCompletionRequest
	| EmbeddingRequest
	| ImageToTextRequest
export type ModelInstanceRequest = ModelRequestMeta & IncomingRequest

export interface ModelOptionsBase {
	url?: string
	file?: string
	location?: string
	engine: BuiltInEngineName | (string & {})
	task: ModelTaskType | (string & {})
	prepare?: 'blocking' | 'async' | 'on-demand'
	minInstances?: number
	maxInstances?: number
	md5?: string
	sha256?: string
}

// this is the internal state/config of the model. its not used in anything user-facing
// so taking some shortcuts here and use one single interface for all engine/task combinations
// this will work until some option clashes between different engines/tasks (same key but different types)
export interface ModelConfig<T extends EngineOptionsBase = EngineOptionsBase>
	extends ModelOptionsBase {
	id: string
	location: string
	task: ModelTaskType | (string & {})
	engine: BuiltInEngineName | (string & {})
	engineOptions?: T
	minInstances: number
	maxInstances: number
	ttl?: number
	contextSize?: number
	grammars?: Record<string, string>
	completionDefaults?: TextCompletionParams
	tools?: Record<string, ToolDefinition>
	preload?: TextCompletionPreloadOptions
}

export interface EngineOptionsBase {
	gpu?: boolean | 'auto' | (string & {})
	gpuLayers?: number
	batchSize?: number
	cpuThreads?: number
}

export interface EngineContext<
	TModelMeta = unknown,
	TOptions extends EngineOptionsBase = EngineOptionsBase,
> {
	config: ModelConfig<TOptions>
	modelMeta?: TModelMeta
	log: Logger
}

export interface EngineTextCompletionArgs<T extends EngineOptionsBase>
	extends EngineContext<T> {
	onChunk?: (chunk: CompletionChunk) => void
	request: TextCompletionRequest
}

export interface EngineChatCompletionArgs<
	T extends EngineOptionsBase = EngineOptionsBase,
> extends EngineContext<T> {
	onChunk?: (chunk: CompletionChunk) => void
	resetContext?: boolean
	request: ChatCompletionRequest
}

export interface EngineEmbeddingArgs<T extends EngineOptionsBase>
	extends EngineContext<T> {
	request: EmbeddingRequest
}

export interface EngineImageToTextArgs<T extends EngineOptionsBase>
	extends EngineContext<T> {
	request: ImageToTextRequest
}

export interface FileDownloadProgress {
	file: string
	loadedBytes: number
	totalBytes: number
}

export interface EngineStartContext {
	pool: ModelPool
	store: ModelStore
}

export interface ModelEngine<
	TInstance = unknown,
	TModel = unknown,
	TOptions extends EngineOptionsBase = EngineOptionsBase,
> {
	autoGpu?: boolean
	start?: (ctx: EngineStartContext) => Promise<void>
	prepareModel: (
		ctx: EngineContext<TOptions>,
		onProgress?: (progress: FileDownloadProgress) => void,
		signal?: AbortSignal,
	) => Promise<TModel>
	createInstance: (
		ctx: EngineContext<TOptions>,
		signal?: AbortSignal,
	) => Promise<TInstance>
	disposeInstance: (instance: TInstance) => Promise<void>
	processChatCompletionTask?: (
		args: EngineChatCompletionArgs<TOptions>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineChatCompletionResult>
	processTextCompletionTask?: (
		args: EngineTextCompletionArgs<TOptions>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineTextCompletionResult>
	processEmbeddingTask?: (
		args: EngineEmbeddingArgs<TOptions>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineEmbeddingResult>
	processImageToTextTask?: (
		args: EngineImageToTextArgs<TOptions>,
		instance: TInstance,
		signal?: AbortSignal,
	) => Promise<EngineImageToTextResult>
}

interface EmbeddingModelOptions extends ModelOptionsBase {
	task: 'embedding'
}

interface TextCompletionModelOptions extends ModelOptionsBase {
	task: 'text-completion'
	contextSize?: number
	grammars?: Record<string, string>
	tools?: Record<string, ToolDefinition>
	completionDefaults?: TextCompletionParams
	preload?: TextCompletionPreloadOptions
}

interface CustomEngineModelOptions extends ModelOptionsBase {
	engine: string
	engineOptions?: EngineOptionsBase
}

interface NodeLlamaCppEmbeddingModelOptions extends EmbeddingModelOptions {
	engine: 'node-llama-cpp'
	engineOptions?: NodeLlamaCppEngineOptions
}

interface NodeLlamaCppTextCompletionModelOptions
	extends TextCompletionModelOptions {
	engine: 'node-llama-cpp'
	engineOptions?: NodeLlamaCppEngineOptions
}

interface GPT4AllTextCompletionModelOptions extends TextCompletionModelOptions {
	engine: 'gpt4all'
	engineOptions?: GPT4AllEngineOptions
}

interface GPT4AllEmbeddingModelOptions extends EmbeddingModelOptions {
	engine: 'gpt4all'
	engineOptions?: GPT4AllEngineOptions
}

interface TransformersJsModelOptions extends ModelOptionsBase {
	engine: 'transformers-js'
	engineOptions?: TransformersJsEngineOptions
}

export type ModelOptions =
	| NodeLlamaCppTextCompletionModelOptions
	| NodeLlamaCppEmbeddingModelOptions
	| GPT4AllTextCompletionModelOptions
	| GPT4AllEmbeddingModelOptions
	| TransformersJsModelOptions
	| CustomEngineModelOptions

export interface EngineEmbeddingResult {
	embeddings: Float32Array[]
	inputTokens: number
}

export interface ChatCompletionResult extends EngineChatCompletionResult {
	id: string
	model: string
}

export interface EngineChatCompletionResult {
	message: AssistantMessage
	finishReason: CompletionFinishReason
	promptTokens: number
	completionTokens: number
	totalTokens: number
}

export interface EngineTextCompletionResult {
	text: string
	finishReason?: CompletionFinishReason
	promptTokens: number
	completionTokens: number
	totalTokens: number
}

export interface EngineImageToTextResult {
	text: string
}
