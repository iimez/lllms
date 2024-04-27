import http from 'node:http'
import { ListenOptions } from 'node:net'
import { existsSync } from 'node:fs'
import express from 'express'
import cors from 'cors'
import { LLMPool, LLMPoolOptions } from './pool.js'
import { LLMInstance } from './instance.js'
import { ModelDownloader } from './downloader.js'
import { createOpenAIRequestHandlers } from './api/openai.js'

export interface LLMServerOptions extends LLMPoolOptions {}

export class LLMServer {
	pool: LLMPool
	loader: ModelDownloader
	ready: Promise<void>
	constructor(opts: LLMServerOptions) {
		this.pool = new LLMPool(opts, this.prepareInstance.bind(this))
		this.ready = this.pool.init()
		this.loader = new ModelDownloader()
	}

	async prepareInstance(instance: LLMInstance, signal?: AbortSignal) {
		const config = instance.config
		if (!existsSync(config.file) && config.url) {
			await this.loader.enqueueDownload(
				{
					file: config.file,
					url: config.url,
				},
				signal,
			)
		}

		// TODO could validate the model file here
	}

	async close() {
		this.pool.queue.clear()
		await this.pool.queue.onIdle()
		await this.pool.dispose()
	}

	getStatusInfo() {
		const pool = this.pool.getStatusInfo()
		return {
			downloads: {
				queue: this.loader.queue.size,
				pending: this.loader.queue.pending,
				tasks: this.loader.tasks,
			},
			pool,
		}
	}
}

export function createOpenAIMiddleware(llmServer: LLMServer) {
	const router = express.Router()
	const requestHandlers = createOpenAIRequestHandlers(llmServer.pool)
	router.get('/v1/models', requestHandlers.listModels)
	router.post('/v1/completions', requestHandlers.completions)
	router.post('/v1/chat/completions', requestHandlers.chatCompletions)
	return router
}

export function createAPIMiddleware(llmServer: LLMServer) {
	const router = express.Router()
	router.get('/', (req, res) => {
		res.json(llmServer.getStatusInfo())
	})
	router.use('/openai', createOpenAIMiddleware(llmServer))
	return router
}

export async function startStandaloneServer(
	serverOpts: LLMServerOptions,
	listenOpts?: ListenOptions,
) {
	const llmServer = new LLMServer(serverOpts)
	const app = createExpressApp(llmServer)
	app.set('json spaces', 2)
	const httpServer = http.createServer(app)

	httpServer.on('close', () => {
		llmServer.close()
	})

	await new Promise<void>((resolve) => {
		httpServer.listen(listenOpts ?? { port: 3000 }, resolve)
	})
	return httpServer
}

export function createExpressApp(llmServer: LLMServer) {
	const app = express()
	app.use(
		cors(),
		express.json({ limit: '50mb' }),
		createAPIMiddleware(llmServer),
	)

	return app
}
