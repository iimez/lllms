import { expect } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { createChatCompletion } from '../../util.js'

export async function runSystemMessageTest(llms: LLMServer) {
	const chat = await createChatCompletion(llms, {
		messages: [
			{
				role: 'system',
				content:
					'Roleplay as Batman. Answer as if you are Batman, never say you are an Assistant or AI model.',
			},
			{
				role: 'user',
				content: 'Hey im Robin! Whats the plan tonight?',
			},
		],
	})
	// console.debug({
	// 	response: chat.result.message.content,
	// })
	expect(chat.result.message.content).toMatch(/Gotham|Joker|Riddler|criminal|Batman/)
}