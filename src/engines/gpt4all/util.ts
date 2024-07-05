import { ChatMessage as GPT4AllChatMessage } from 'gpt4all'
import { ChatMessage } from '#lllms/types/index.js'
import { flattenMessageTextContent } from '#lllms/lib/flattenMessageTextContent.js'
import { calculateFileChecksum } from '#lllms/lib/calculateFileChecksum.js'

export function createChatMessageArray(
	messages: ChatMessage[],
): GPT4AllChatMessage[] {
	const chatMessages: GPT4AllChatMessage[] = []
	let systemPrompt: string | undefined
	for (const message of messages) {
		if (message.role === 'user' || message.role === 'assistant') {
			chatMessages.push({
				role: message.role,
				content: flattenMessageTextContent(message.content),
			})
		} else if (message.role === 'system') {
			if (systemPrompt) {
				systemPrompt += '\n\n' + message.content
			} else {
				systemPrompt = flattenMessageTextContent(message.content)
			}
		}
	}
	if (systemPrompt) {
		chatMessages.unshift({
			role: 'system',
			content: systemPrompt,
		})
	}
	return chatMessages
}

export async function verifyModelFile(location: string, md5: string) {
	const fileHash = await calculateFileChecksum(location, 'md5')
	if (fileHash !== md5) {
		throw new Error(
			`Model md5 checksum mismatch: expected ${md5} got ${fileHash} for ${location}`,
		)
	}
}