import http from 'node:http'
import { ListenOptions } from 'node:net'
import express from 'express'
import cors from 'cors'
import { createOpenAIRequestHandlers } from '#lllms/api/openai/index.js'
import { createAPIMiddleware } from '#lllms/api/v1/index.js'
import { LogLevel } from '#lllms/lib/logger.js'
import { LLMServer, LLMServerOptions } from '#lllms/server.js'

export function createOpenAIMiddleware(llmServer: LLMServer) {
	const router = express.Router()
	const requestHandlers = createOpenAIRequestHandlers(llmServer)
	router.get('/v1/models', requestHandlers.models)
	router.post('/v1/completions', requestHandlers.completions)
	router.post('/v1/chat/completions', requestHandlers.chatCompletions)
	router.post('/v1/embeddings', requestHandlers.embeddings)
	return router
}

export function createExpressMiddleware(llmServer: LLMServer) {
	const router = express.Router()
	router.get('/', (req, res) => {
		res.json(llmServer.getStatus())
	})
	router.use('/openai', createOpenAIMiddleware(llmServer))
	router.use('/llama', createAPIMiddleware(llmServer))
	return router
}

export interface StandaloneServerOptions extends LLMServerOptions {
	listen?: ListenOptions
	logLevel?: LogLevel
}

export async function serveLLMs(options: StandaloneServerOptions) {
	const { listen, ...serverOpts } = options
	const listenOpts = listen ?? { port: 3000 }
	const llmServer = new LLMServer(serverOpts)
	await llmServer.start()
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
	await new Promise<void>((resolve) => {
		httpServer.listen(listenOpts, resolve)
	})
	return httpServer
}
