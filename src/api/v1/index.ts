import { IncomingMessage, ServerResponse } from 'node:http'
import express from 'express'
import { LLMPool } from '#lllms/pool.js'
import { LLMServer } from '#lllms/server.js'

export function createAPIMiddleware(server: LLMServer) {
	const router = express.Router()
	
	router.use((req, res, next) => {
		console.debug('API call', {
			method: req.method,
			url: req.url,
			body: req.body,
		})
		next()
	})
	return router
	// return async (req: IncomingMessage, res: ServerResponse) => {
		
	// 	let args: any
		
	// 	try {
	// 		const body = await parseJSONRequestBody(req)
	// 		args = body
	// 	} catch (e) {
	// 		console.error(e)
	// 		res.writeHead(400, { 'Content-Type': 'application/json' })
	// 		res.end(JSON.stringify({ error: 'Invalid request' }))
	// 		return
	// 	}
		
	// 	console.debug('Handler', JSON.stringify(args, null, 2))
	// 	res.writeHead(200, { 'Content-Type': 'application/json' })
	// 	res.end(JSON.stringify({ message: 'Hello' }))
		
	// }
}