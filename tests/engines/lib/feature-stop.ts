import { expect } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { createChatCompletion, createCompletion } from '../../util.js'

export async function runStopTriggerTest(llms: LLMServer) {
	const chat = await createChatCompletion(llms, {
		messages: [
			{
				role: 'user',
				content: "This is a test. Please only answer with 'OK'.",
			},
		],
		stop: ['OK'],
		maxTokens: 10,
	})
	expect(chat.result.finishReason).toBe('stopGenerationTrigger')
	expect(chat.result.message.content).toBe('')
	const completion = await createCompletion(llms, {
		prompt: "Let's count to four. One, two,",
		stop: [' three'],
		maxTokens: 10,
	})
	expect(completion.result.finishReason).toBe('stopGenerationTrigger')
	expect(completion.result.text).toBe('')
}
