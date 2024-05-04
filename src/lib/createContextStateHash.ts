import crypto from 'node:crypto'
import { ChatMessage } from '#lllms/types/index.js'

interface ContextStateData {
	messages?: ChatMessage[]
	prompt?: string
	systemPrompt?: string
}
export function createContextStateHash(
	state: ContextStateData,
	dropLastMessage: boolean = false,
): string {
	const messages = state.messages ? [...state.messages] : []
	if (dropLastMessage && messages.length > 1) {
		messages.pop()
	}
	const systemPromptData = state.systemPrompt || ''
	const messagesData = messages
		.map((message) => message.role + message.content)
		.join('\n')
	const promptData = state.prompt || ''
	return crypto
		.createHash('sha1')
		.update(systemPromptData + messagesData + promptData)
		.digest('hex')
}
