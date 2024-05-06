import { expect } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { createChatCompletion } from '../../util.js'

export async function runTokenBiasTest(llms: LLMServer) {
	const unbiasedChat = await createChatCompletion(llms, {
		messages: [
			{
				role: 'user',
				content: 'Please finish my sentence: "Once upon a..."',
			},
		],
	})
	// console.debug({
	// 	unbiasedResponse: unbiasedChat.result.message.content,
	// })
	expect(unbiasedChat.result.message.content).toMatch(/time/)
	const biasedChat = await createChatCompletion(llms, {
		tokenBias: {
			'time': -100,
			'a time...': -100,
		},
		messages: [
			{
				role: 'user',
				content: 'Please finish my sentence: "Once upon a..."',
			},
		],
	})
	// console.debug({
	// 	biasedResponse: biasedChat.result.message.content,
	// })
	expect(biasedChat.result.message.content).not.toMatch(/time/)
}