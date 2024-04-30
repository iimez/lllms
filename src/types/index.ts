import type { EngineType, EngineInstance } from '../engines'
import type { Logger } from '../util/log.js'
export interface CompletionChunk {
	tokens: number[]
	text: string
}

export type CompletionFinishReason =
	| 'maxTokens'
	| 'functionCall'
	| 'eogToken'
	| 'stopGenerationTrigger'

export interface ChatMessage {
	role: 'user' | 'assistant' | 'system'
	content: string
}

export type ChatTemplateFormat = 'chatml' | 'llama3' | 'alpaca' |'phi'

export interface CompletionRequestBase {
	model: string
	temperature?: number
	stream?: boolean
	maxTokens?: number
	seed?: number
	stop?: string[]
	frequencyPenalty?: number
	presencePenalty?: number
	topP?: number
	minP?: number
	topK?: number
}

export interface CompletionRequest extends CompletionRequestBase {
	prompt?: string
}

export interface ChatCompletionRequest extends CompletionRequestBase {
	systemPrompt?: string
	messages: ChatMessage[]
	templateFormat?: ChatTemplateFormat
}

export interface EngineCompletionContext extends EngineContext {
	onChunk?: (chunk: CompletionChunk) => void
	resetContext?: boolean
}

export interface EngineContext {
	signal?: AbortSignal
	logger: Logger
	instance: string
}

export interface EngineChatCompletionResult {
	message: ChatMessage
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

export interface ChatCompletionResult extends EngineChatCompletionResult {
	id: string
	model: string
}

export interface LLMEngine {
	loadInstance: (
		config: LLMConfig,
		ctx: EngineContext,
	) => Promise<EngineInstance>
	disposeInstance: (instance: EngineInstance) => Promise<void>
	processChatCompletion: (
		instance: EngineInstance,
		req: ChatCompletionRequest,
		ctx: EngineCompletionContext,
	) => Promise<EngineChatCompletionResult>
	processCompletion: (
		instance: EngineInstance,
		req: CompletionRequest,
		ctx: EngineCompletionContext,
	) => Promise<EngineCompletionResult>
}

export interface LLMOptions {
	gpu?: boolean | string
	url?: string
	file?: string
	engine: EngineType
	minInstances?: number
	maxInstances?: number
	templateFormat?: ChatTemplateFormat
}


export interface LLMConfig extends LLMOptions {
	name: string
	file: string
	engine: EngineType
}