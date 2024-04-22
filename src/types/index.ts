
type CompletionToken = any

export interface ChatMessage {
	role: 'user' | 'assistant' | 'system'
	content: string
}

export interface ChatCompletionArgs {
	model: string
	messages: ChatMessage[]
	systemPrompt?: string
	resetContext?: boolean
	temperature?: number
	stream?: boolean
	maxTokens?: number
	seed?: number
	frequencyPenalty?: number
	presencePenalty?: number
	topP?: number
	onToken?: (token: CompletionToken) => void
}

export interface EngineChatCompletionResult {
	message: ChatMessage
	finishReason: "maxTokens" | "functionCall" | "eosToken" | "stopGenerationTrigger"
	promptTokens: number
	completionTokens: number
	totalTokens: number
}

export interface ChatCompletionResult extends EngineChatCompletionResult {
	id: string
	model: string
}
