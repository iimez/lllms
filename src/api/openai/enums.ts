import { CompletionFinishReason, ChatMessage } from '#lllms/types/index.js'
import OpenAI from 'openai'

export const finishReasonMap: Record<CompletionFinishReason, OpenAI.ChatCompletion.Choice['finish_reason']> = {
	maxTokens: 'length',
	toolCalls: 'tool_calls',
	eogToken: 'stop',
	stopTrigger: 'stop',
	timeout: 'stop',
	cancel: 'stop',
	abort: 'stop',
} as const

export const messageRoleMap: Record<OpenAI.ChatCompletionMessageParam['role'], ChatMessage['role']> = {
	user: 'user',
	system: 'system',
	assistant: 'assistant',
	tool: 'tool',
	function: 'tool',
}