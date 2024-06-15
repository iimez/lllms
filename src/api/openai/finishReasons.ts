import { CompletionFinishReason } from '#lllms/types/index.js'
import OpenAI from 'openai'

export const finishReasons: Record<CompletionFinishReason, OpenAI.ChatCompletion.Choice['finish_reason']> = {
	maxTokens: 'length',
	functionCall: 'tool_calls',
	eogToken: 'stop',
	stopTrigger: 'stop',
	timeout: 'stop',
	cancel: 'stop',
	abort: 'stop',
} as const
