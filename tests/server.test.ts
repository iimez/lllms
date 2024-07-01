import { describe, it, expect, beforeAll } from 'vitest'
import request from 'supertest'
import express, { Express } from 'express'
import { ModelServer, ModelServerOptions } from '#lllms/server.js'
import { createExpressMiddleware } from '#lllms/http.js'

const testModel = 'phi3-mini-4k'

const testConfig: ModelServerOptions = {
	concurrency: 1,
	models: {
		[testModel]: {
			url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
			task: 'text-completion',
			engine: 'gpt4all',
			minInstances: 1,
		},
	},
}

describe('Express App', () => {
	let app: Express
	let llmServer: ModelServer

	beforeAll(async () => {
		llmServer = new ModelServer(testConfig)
		app = express()
		app.use(express.json(), createExpressMiddleware(llmServer))
	})

	it('Responds to requests before starting', async () => {
		const res = await request(app).get('/')
		expect(res.status).toBe(200)
		expect(res.body).toMatchObject({
			downloads: { queue: 0, pending: 0, tasks: [] },
			pool: { processing: 0, waiting: 0, instances: {} },
		})
	})

	it('Starts up without errors', async () => {
		await llmServer.start()
		// TODO anything useful to assert here?
	})

	it('Has an instance of the model ready', async () => {
		const res = await request(app).get('/')
		expect(res.status).toBe(200)
		expect(Object.keys(res.body.pool.instances).length).toBe(1)
		const instanceKey = Object.keys(res.body.pool.instances)[0] as string
		const instance = res.body.pool.instances[instanceKey]
		expect(instance.model).toBe(testModel)
	})
})
