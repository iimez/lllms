import crypto from 'node:crypto'
import { ChatMessage } from '#lllms/types/index.js'
import { flattenMessageTextContent } from './flattenMessageTextContent.js'

export interface CalculateContextIdentityOptions {
	messages?: ChatMessage[]
	text?: string
	dropLastMessage?: boolean
}

export function calculateContextIdentity({
	messages,
	text,
	dropLastMessage = false,
}: CalculateContextIdentityOptions): string {

	if (text) {
		return text
	}
	
	if (messages?.length) {
		const filteredMessages = messages.filter((message, i) => {
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
		if (dropLastMessage && filteredMessages.length > 1) {
			if (filteredMessages[filteredMessages.length - 1].role !== 'user') {
				console.warn('Dropping last message that is not a user message. This should not happen.')
			}
			filteredMessages.pop()
		}
		// we dont wanna json stringify because this would make message key order significant
		const serializedMessages = filteredMessages
			.map((message) => {
				return message.role + ': ' + flattenMessageTextContent(message.content)
			})
			.join('\n')
			const contextIdentity = crypto
			.createHash('sha1')
			.update(serializedMessages)
			.digest('hex')
		return contextIdentity
	}
	
	return ''
}
