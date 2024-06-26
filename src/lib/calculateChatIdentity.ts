import crypto from 'node:crypto'
import { ChatMessage } from '#lllms/types/index.js'
import { flattenMessageTextContent } from './flattenMessageTextContent.js'

export function calculateChatIdentity(
	inputMessages: ChatMessage[],
	dropLastMessage: boolean = false,
): string {
	let inputData = ''
	const messages = inputMessages.filter((message, i) => {
		// remove all but the leading system message
		if (message.role === 'system' && i !== 0) {
			return false
		}
		if (message.role === 'tool') {
			return false
		}
		const textContent = flattenMessageTextContent(message.content)
		return !!textContent
	})
	if (dropLastMessage && messages.length > 1) {
		if (messages[messages.length - 1].role !== 'user') {
			console.warn('Dropping last message that is not a user message. This should not happen.')
		}
		messages.pop()
	}
	// we dont wanna json stringify because this would make message key order significant
	const serializedMessages = messages
		.map((message) => {
			return message.role + ': ' + flattenMessageTextContent(message.content)
		})
		.join('\n')
	inputData += serializedMessages
	const contextIdentity = crypto
		.createHash('sha1')
		.update(inputData)
		.digest('hex')
	return contextIdentity
}
