import http from 'node:http'
import { ListenOptions } from 'node:net'
import { promises as fs, existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import express from 'express'
import cors from 'cors'
import { LLMPool } from './pool.js'
import { LLMInstance } from './instance.js'
import { ModelDownloader } from './downloader.js'
import { LLMOptions } from './types/index.js'
import { createOpenAIRequestHandlers } from './api/openai.js'
import { resolveModelConfig } from './util/resolveModelConfig.js'


export interface LLMServerOptions {
	concurrency?: number
	modelsDir?: string
	models: Record<string, LLMOptions>
}

export class LLMServer {
	pool: LLMPool
	loader: ModelDownloader
	modelsDir: string
	constructor(opts: LLMServerOptions) {
		this.modelsDir = opts.modelsDir || path.resolve(os.homedir(), '.cache/lllms')
		const poolModels = resolveModelConfig(opts.models, this.modelsDir)
		this.pool = new LLMPool({
			concurrency: opts.concurrency ?? 1,
			models: poolModels
		}, this.prepareInstance.bind(this))
		this.loader = new ModelDownloader()
	}
	
	async start() {
		await fs.mkdir(this.modelsDir, { recursive: true })
		await this.pool.init()
	}

	// gets called by the pool right before a new instance is created
	async prepareInstance(instance: LLMInstance, signal?: AbortSignal) {
		// make sure the model files exists, download if possible.
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
		if (!existsSync(config.file)) {
			throw new Error(`Model file not found: ${config.file}`)
		}

		// TODO good place to validate the model file, if necessary
	}

	async stop() {
		// TODO need to do more cleanup here
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

export function createExpressMiddleware(llmServer: LLMServer) {
	const router = express.Router()
	router.get('/', (req, res) => {
		res.json(llmServer.getStatusInfo())
	})
	router.use('/openai', createOpenAIMiddleware(llmServer))
	return router
}

export interface StandaloneServerOptions extends LLMServerOptions {
	listen?: ListenOptions
}

export async function serveLLMs(
	opts: StandaloneServerOptions
) {
	const { listen, ...serverOpts } = opts
	const listenOpts = listen ?? { port: 3000 }
	const llmServer = new LLMServer(serverOpts)
	
	const app = express()
	app.use(
		cors(),
		express.json({ limit: '50mb' }),
		createExpressMiddleware(llmServer),
	)

	app.set('json spaces', 2)
	const httpServer = http.createServer(app)

	httpServer.on('close', () => {
		llmServer.stop()
	})

	const initPromise = llmServer.start()
	const listenPromise = new Promise<void>((resolve) => {
		httpServer.listen(listenOpts, resolve)
	})
	await listenPromise
	// await Promise.all([listenPromise, llmServer.start()])
	return httpServer
}
