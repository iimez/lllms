import type { EngineType, EngineInstance } from '../engines'
export interface CompletionChunk {
	tokenId: number
	token: string
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
}

export interface CompletionRequest extends CompletionRequestBase {
	prompt?: string
}

export interface ChatCompletionRequest extends CompletionRequestBase {
	// model: string
	systemPrompt?: string
	messages: ChatMessage[]
	templateFormat?: ChatTemplateFormat
	// prompt?: string
	// systemPrompt?: string
	// temperature?: number
	// stream?: boolean
	// maxTokens?: number
	// seed?: number
	// stop?: string[]
	// frequencyPenalty?: number
	// presencePenalty?: number
	// topP?: number
	// templateFormat?: ChatTemplateFormat
}

export interface GenerationArgs {
	onChunk?: (chunk: CompletionChunk) => void
	resetContext?: boolean
	signal?: AbortSignal
}

export interface EngineChatCompletionResult {
	message: ChatMessage
	finishReason?: CompletionFinishReason
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
		signal?: AbortSignal,
	) => Promise<EngineInstance>
	disposeInstance: (instance: EngineInstance) => Promise<void>
	processChatCompletion: (
		instance: EngineInstance,
		completionArgs: ChatCompletionRequest,
		processingArgs?: GenerationArgs,
	) => Promise<EngineChatCompletionResult>
	processCompletion: (
		instance: EngineInstance,
		completionArgs: CompletionRequest,
		processingArgs?: GenerationArgs,
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