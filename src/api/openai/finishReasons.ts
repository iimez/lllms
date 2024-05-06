import { CompletionFinishReason } from '#lllms/types/index.js'
import OpenAI from 'openai'

export const finishReasons: Record<CompletionFinishReason, OpenAI.ChatCompletion.Choice['finish_reason']> = {
	maxTokens: 'length',
	functionCall: 'function_call', // TODO tool_calls
	eogToken: 'stop',
	stopGenerationTrigger: 'stop',
	timeout: 'stop',
	cancelled: 'stop',
} as const
