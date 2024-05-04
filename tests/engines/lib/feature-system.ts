import { expect } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { createChatCompletion, parseInstanceId } from '../../util.js'

export async function runSystemPromptTest(llms: LLMServer) {
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
	expect(chat.result.message.content).toMatch(/Gotham|Batman/)
}