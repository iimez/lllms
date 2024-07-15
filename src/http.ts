import http from 'node:http'
import { ListenOptions } from 'node:net'
import express from 'express'
import cors from 'cors'
import { createOpenAIRequestHandlers } from '#lllms/api/openai/index.js'
import { createAPIMiddleware } from '#lllms/api/v1/index.js'
import { LogLevel } from '#lllms/lib/logger.js'
import { ModelServer, ModelServerOptions, startModelServer } from '#lllms/server.js'

export function createOpenAIMiddleware(modelServer: ModelServer) {
	const router = express.Router()
	const requestHandlers = createOpenAIRequestHandlers(modelServer)
	router.get('/v1/models', requestHandlers.models)
	router.post('/v1/completions', requestHandlers.completions)
	router.post('/v1/chat/completions', requestHandlers.chatCompletions)
	router.post('/v1/embeddings', requestHandlers.embeddings)
	return router
}

export function createExpressMiddleware(modelServer: ModelServer) {
	const router = express.Router()
	router.get('/', (req, res) => {
		res.json(modelServer.getStatus())
	})
	router.use('/openai', createOpenAIMiddleware(modelServer))
	router.use('/llama', createAPIMiddleware(modelServer))
	return router
}

export interface ModelHTTPServerOptions extends ModelServerOptions {
	listen?: ListenOptions
	logLevel?: LogLevel
}

export class ModelHTTPServer {
	httpServer: http.Server
	modelServer: ModelServer
	listenOptions: ListenOptions
	
	constructor(options: ModelHTTPServerOptions) {
		const { listen, ...modelServerOpts } = options
		this.modelServer = new ModelServer(modelServerOpts)
		this.listenOptions = listen ?? { port: 3000 }
		const app = express()
		app.use(
			cors(),
			express.json({ limit: '50mb' }),
			createExpressMiddleware(this.modelServer),
		)
	
		app.set('json spaces', 2)
		this.httpServer = http.createServer(app)
	}
	
	async start() {
		await this.modelServer.start()
		await new Promise<void>((resolve) => {
			this.httpServer.listen(this.listenOptions, resolve)
		})
	}
	
	async close() {
		this.httpServer.close()
		await this.modelServer.stop()
	}
}

export async function startHTTPServer(options: ModelHTTPServerOptions) {
	const server = new ModelHTTPServer(options)
	await server.start()
	return server
}