import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { ModelServer } from '#lllms/server.js'
import { createChatCompletion } from './util.js'

suite('basic', () => {
	const llms = new ModelServer({
		log: 'debug',
		models: {
			test: {
				task: 'text-completion',
				url: 'https://huggingface.co/mradermacher/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf',
				sha256:
					'8729adfbc1cdaf3229ddeefab2b58ffdc78dbdb4d92234bcd5980c53f12fad15',
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
		expect(chat.result.message.content.length).toBeGreaterThan(0)
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
		expect(chat1.result.message.content.length).toBeGreaterThan(0)
		const chat2 = await createChatCompletion(llms, {
			temperature: 1,
			messages: [
				{
					role: 'user',
					content: 'Tell me a story, but just its title.',
				},
			],
		})
		expect(chat2.result.message.content.length).toBeGreaterThan(0)
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
		expect(results.length).toBe(10)
		expect(results.every((r) => r.result.message.content.length > 0)).toBe(true)
	})
})

suite('gpu', () => {
	const llms = new ModelServer({
		log: 'debug',
		models: {
			gpt4all: {
				url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
				task: 'text-completion',
				md5: 'f8347badde9bfc2efbe89124d78ddaf5',
				engine: 'gpt4all',
				device: { gpu: true },
			},
			'node-llama-cpp': {
				url: 'hhttps://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf',
				task: 'text-completion',
				sha256: '39458b227a4be763b7eb39d306d240c3d45205e3f8b474ec7bdca7bba0158e69',
				engine: 'node-llama-cpp',
				device: { gpu: true },
			},
		},
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('gpu completion', async () => {
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

	test('switch to different gpu model when necessary', async () => {
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
	
	test('handle simultaneous requests to two gpu models', async () => {
		const [chat1, chat2] = await Promise.all([
			createChatCompletion(llms, {
				model: 'node-llama-cpp',
				messages: [
					{
						role: 'user',
						content: 'Tell me a story, but just its title.',
					},
				],
			}),
			createChatCompletion(llms, {
				model: 'gpt4all',
				messages: [
					{
						role: 'user',
						content: 'Tell me a story, but just its title.',
					},
				],
			}),
		])
		expect(chat1.device).toBe('gpu')
		expect(chat2.device).toBe('gpu')
	})
})
