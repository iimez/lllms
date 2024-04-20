import http from 'node:http'
import { URL } from 'node:url'
import { LLMPool, LLMPoolOptions } from './pool.js'
import { createOpenAIRequestHandlers } from './api/openai.js'

export type InferenceServerOptions = LLMPoolOptions

export async function createServer(opts: InferenceServerOptions) {
	// "const inferServer = new InferenceServer(config)" ?
	const pool = new LLMPool(opts)
	const initPromise = pool.init()

	const requestHandlers = createOpenAIRequestHandlers(pool)

	const server = http.createServer((req, res) => {
		try {
			if (req.url) {
				const parsedUrl = new URL(req.url, 'http://localhost')
				if (parsedUrl.pathname === '/v1/models') {
					requestHandlers.listModels(req, res)
					return
				}
				if (parsedUrl.pathname === '/v1/completions') {
					requestHandlers.completions(req, res)
					return
				}
				if (parsedUrl.pathname === '/v1/chat/completions') {
					requestHandlers.chatCompletions(req, res)
					return
				}
				if (parsedUrl.pathname === '/') {
					res.writeHead(200, { 'Content-Type': 'application/json' })
					res.end(JSON.stringify(pool.getStatusInfo(), null, 2))
					return
				}
			}

			res.writeHead(404, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Path not found' }))
		} catch (e) {
			console.error(e)
			res.writeHead(500, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Internal server error' }))
		}
	})

	server.on('close', () => {
		pool.queue.clear()
		pool.queue.onIdle().then(() => {
			pool.dispose()
		})
	})

	return { server, initPromise }
}
