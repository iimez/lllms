import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { Server } from 'node:http'
import request from 'supertest'
import { createLLMServer, LLMServerOptions } from '../src/server.js'
import { LLMPoolConfig } from '../src/pool.js'

const testConfig: LLMServerOptions = {
	concurrency: 1,
	models: {
		'orca-3b': {
			url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
			minInstances: 1,
			maxInstances: 2,
			engine: 'gpt4all',
		},
	},
}

suite('Cache Tests', () => {
	let server: Server

	beforeAll(async () => {
		const res = await createLLMServer(testConfig)
		server = res.server
		server.listen()
		await res.initPromise
	})

	afterAll(async () => {
		server.close()
	})
	
	test('Cache hit', async () => {
		// TODO
	})

	test('Cache leakage', async () => {
		const messagesA = [
			{ role: 'user', content: 'The secret is 34572. Please remember.' },
		]
		const requestA = await request(server)
			.post('/v1/chat/completions')
			.send({
				model: 'orca-3b',
				temperature: 0,
				messages: messagesA,
			})
		messagesA.push(requestA.body.choices[0].message)
		// console.debug('a res', requestA.body.choices[0].message.content)
		const requestB = await request(server)
			.post('/v1/chat/completions')
			.send({
				model: 'orca-3b',
				temperature: 0,
				messages: [
					{ role: 'user', content: 'What is the secret?' },
				],
			})
		// TODO assert request B doesnt know the secret
		messagesA.push({
			role: 'user',
			content: 'What is the secret?',
		})
		const requestA2 = await request(server)
			.post('/v1/chat/completions')
			.send({
				model: 'orca-3b',
				temperature: 0,
				messages: messagesA,
			})
		// TODO assert request A2 does know the secret
	})
	
	// test('POST /v1/chat/completions should return status 200', async () => {
	// 	const response = await request(server)
	// 		.post('/v1/chat/completions')
	// 		.send({
	// 			model: 'orca-3b',
	// 			temperature: 0,
	// 			messages: [
	// 				{ role: 'user', content: 'This is a test. Just answer with "Test".' },
	// 			],
	// 		})
	// 	expect(response.status).toBe(200)
	// 	expect(response.body.choices[0].message.content).toContain('Test')
	// })
})
