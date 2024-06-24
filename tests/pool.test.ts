import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { LLMOptions } from '#lllms/types/index.js'
import { createChatCompletion } from './util.js'

const models: Record<string, LLMOptions> = {
	'test': {
		task: 'inference',
		url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
		sha256: '1977ae6185ef5bc476e27db85bb3d79ca4bd87e7b03399083c297d9c612d334c',
		engine: 'node-llama-cpp',
	},
}

suite('pool', () => {
	const llms = new LLMServer({ models })

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('does a completion', async () => {
		const chat = await createChatCompletion(llms, {
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			]
		})
		console.log({
			res: chat.result.message,
		})
	})

	test('does a second completion', async () => {
		const chat = await createChatCompletion(llms, {
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			]
		})
		console.log({
			res: chat.result.message,
		})
	})
	
	test('does a third completion', async () => {
		const chat = await createChatCompletion(llms, {
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			]
		})
		console.log({
			res: chat.result.message,
		})
	})

})
