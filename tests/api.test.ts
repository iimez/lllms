import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { Server } from 'node:http'
import request from 'supertest'
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

suite('API Integration Tests', () => {
	let server: Server

	beforeAll(async () => {
		server = await createServer(testPoolConfig)
		server.listen()
	})

	afterAll(async () => {
		server.close()
	})

	test('POST /v1/chat/completions should return status 200', async () => {
		const response = await request(server)
			.post('/v1/chat/completions')
			.send({
				model: 'orca-mini-3b-gguf2-q4_0',
				temperature: 0,
				messages: [
					{ role: 'user', content: 'This is a test. Just answer with "Test".' },
				],
			})
		expect(response.status).toBe(200)
		expect(response.body.choices[0].message.content).toContain('Test')
	})
})
