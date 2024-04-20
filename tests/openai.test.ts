import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { Server } from 'node:http'
import OpenAI from 'openai'
import { createServer, InferenceServerOptions } from '../src/server.js'

const testConfig: InferenceServerOptions = {
	concurrency: 1,
	models: {
		'orca:3b': {
			url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
			preload: true,
			engine: 'gpt4all',
		},
	},
}

suite('OpenAI Client Integration Tests', () => {
	let server: Server
	const openai = new OpenAI({
		baseURL: 'http://localhost:3000/v1/',
		apiKey: '123',
	})

	beforeAll(async () => {
		const res = await createServer(testConfig)
		server = res.server
		server.listen(3000)
		await res.initPromise
	})

	afterAll(async () => {
		server.close()
	})

	test('openai.chat.completions.create', async () => {
		const chatCompletion = await openai.chat.completions.create({
			model: 'orca:3b',
			temperature: 0,
			messages: [
				{ role: 'user', content: 'This is a test. Just answer with "Test".' },
			],
		})
		// console.debug('chatCompletion:', chatCompletion)
		expect(chatCompletion.choices[0].message.content).toContain('Test')
	})
})
