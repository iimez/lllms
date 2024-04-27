import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import request from 'supertest'
import express, {Express } from 'express'
import { createExpressApp, LLMServerOptions, LLMServer } from '../src/server.js'

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

suite('Express App', () => {
	let app: Express

	beforeAll(async () => {
		const llmServer = new LLMServer(testConfig)
		app = createExpressApp(llmServer)
		await llmServer.ready
	})
	
	test('Test the root', async () => {
		const res = await request(app)
			.get('/')
		expect(res.status).toBe(200)
	})

})
