import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { ModelServer } from '#lllms/server.js'
import { ModelOptions } from '#lllms/types/index.js'
import { createChatCompletion } from './util.js'

suite('basic', () => {
	const llms = new ModelServer({
		models: {
			test: {
				task: 'text-completion',
				url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
				sha256:
					'1977ae6185ef5bc476e27db85bb3d79ca4bd87e7b03399083c297d9c612d334c',
				engine: 'node-llama-cpp',
			},
		},
	})

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
			],
		})
		console.log({
			res: chat.result.message,
		})
	})

	test('does two consecutive completions', async () => {
		const chat1 = await createChatCompletion(llms, {
			temperature: 1,
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		console.log({
			res: chat1.result.message,
		})
		
		const chat2 = await createChatCompletion(llms, {
			temperature: 1,
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		console.log({
			id: chat2.task.id,
			response: chat2.result.message.content,
		})
	})

	test('handles 10 simultaneous completion requests', async () => {
		const results = await Promise.all(
			Array.from({ length: 10 }, () =>
				createChatCompletion(llms, {
					temperature: 1,
					messages: [
						{
							role: 'user',
							content: 'Tell me a story, but just its title.',
						},
					],
				}),
			),
		)
		console.log({
			res: results.map((r) => {
				return {
					id: r.task.id,
					response: r.result.message.content,
				}
			}),
		})
	})
})

suite('gpu', () => {
	const llms = new ModelServer({
		log: 'debug',
		models: {
			gpt4all: {
				url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
				task: 'text-completion',
				// md5: 'f8347badde9bfc2efbe89124d78ddaf5',
				engine: 'gpt4all',
				engineOptions: {
					gpu: true,
					batchSize: 512,
				},
			},
			'node-llama-cpp': {
				url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
				task: 'text-completion',
				// sha256: '8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef',
				engine: 'node-llama-cpp',
				engineOptions: {
					gpu: true,
					batchSize: 512,
				},
			},
		},
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('does a gpu completion', async () => {
		const chat = await createChatCompletion(llms, {
			model: 'gpt4all',
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		expect(chat.device).toBe('gpu')
	})

	test('loads different gpu model when necessary', async () => {
		const chat = await createChatCompletion(llms, {
			model: 'node-llama-cpp',
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		expect(chat.device).toBe('gpu')
	})
})
