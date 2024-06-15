import { suite, it, expect, beforeAll, afterAll } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { LLMOptions } from '#lllms/types/index.js'
import { createChatCompletion } from './util.js'

const models: Record<string, LLMOptions> = {
	'gpt4all': {
		url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
		task: 'inference',
		// md5: 'f8347badde9bfc2efbe89124d78ddaf5',
		engine: 'gpt4all',
		engineOptions: {
			gpu: true,
			batchSize: 512,
		},
	},
	'node-llama-cpp': {
		url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
		task: 'inference',
		// sha256: '8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef',
		engine: 'node-llama-cpp',
		engineOptions: {
			gpu: true,
			batchSize: 512,
		},
	},
}

suite('GPU', () => {
	const llms = new LLMServer({ models })

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	it('does a gpu completion', async () => {
		const chat = await createChatCompletion(llms, {
			model: 'gpt4all',
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			]
		})
		expect(chat.device).toBe('gpu')
	})

	it('loads different gpu model when necessary', async () => {
		const chat = await createChatCompletion(llms, {
			model: 'node-llama-cpp',
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			]
		})
		expect(chat.device).toBe('gpu')
	})
	


})
