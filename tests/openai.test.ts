import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { Server } from 'node:http'
import OpenAI from 'openai'
import { createServer } from '../src/server.js'
import { LLMPoolConfig } from '../src/pool.js'

const testPoolConfig: LLMPoolConfig = {
	concurrency: 1,
	variants: [
		{
			model: 'orca-mini-3b-gguf2-q4_0',
			preload: true,
		},
	],
}

suite('OpenAI Client Integration Tests', () => {
	let server: Server
	const openai = new OpenAI({
		baseURL: 'http://localhost:3000/v1/',
		apiKey: '123',
	})

	beforeAll(async () => {
		server = await createServer(testPoolConfig)
		server.listen(3000)
	})

	afterAll(async () => {
		server.close()
	})

	test('openai.chat.completions.create', async () => {
		const chatCompletion = await openai.chat.completions.create({
			model: 'orca-mini-3b-gguf2-q4_0',
			temperature: 0,
			messages: [
				{ role: 'user', content: 'This is a test. Just answer with "Test".' },
			],
		})
		// console.debug('chatCompletion:', chatCompletion)
		expect(chatCompletion.choices[0].message.content).toContain('Test')
	})
})
