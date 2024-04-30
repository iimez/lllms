import { CompletionFinishReason } from '../../types/index.js'

export const finishReasons: Record<CompletionFinishReason, string> = {
	maxTokens: 'length',
	functionCall: 'function_call', // TODO tool_calls
	eogToken: 'stop',
	stopGenerationTrigger: 'stop',
} as const
