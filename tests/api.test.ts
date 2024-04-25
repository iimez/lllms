import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { Server } from 'node:http'
import request from 'supertest'
import { createServer, LLMServerOptions } from '../src/server.js'
import { LLMPoolConfig } from '../src/pool.js'

const testConfig: LLMServerOptions = {
	concurrency: 1,
	models: {
		'orca-3b': {
			url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
			minInstances: 1,
			engine: 'gpt4all',
		},
	},
}

suite('API Integration Tests', () => {
	let server: Server

	beforeAll(async () => {
		const res = await createServer(testConfig)
		server = res.server
		server.listen()
		await res.initPromise
	})

	afterAll(async () => {
		server.close()
	})

	test('POST /v1/chat/completions should return status 200', async () => {
		const response = await request(server)
			.post('/v1/chat/completions')
			.send({
				model: 'orca-3b',
				temperature: 0,
				messages: [
					{ role: 'user', content: 'This is a test. Just answer with "Test".' },
				],
			})
		expect(response.status).toBe(200)
		expect(response.body.choices[0].message.content).toContain('Test')
	})

        test('openai.models.list ', async () => {
            const response = await request(server).post("/v1/models").send();
            expect(response.status).toBe(200)
        })
})
