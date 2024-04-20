import { IncomingMessage, ServerResponse } from 'node:http'
import { LLMPool } from '../pool.js'

function parseJSONRequestBody(req: IncomingMessage): Promise<any> {
	return new Promise((resolve, reject) => {
		let body = ''

		req.on('data', (chunk) => {
			body += chunk.toString()
		})

		req.on('end', () => {
			try {
				const data = JSON.parse(body)
				resolve(data)
			} catch (error) {
				reject(error)
			}
		})
	})
}

// https://platform.openai.com/docs/api-reference/chat/create
function createChatCompletionHandler(pool: LLMPool) {
	return async (req: IncomingMessage, res: ServerResponse) => {

		let model, messages

		try {
			const reqBody = await parseJSONRequestBody(req)
			model = reqBody.model
			messages = reqBody.messages
		} catch (e) {
			console.error(e)
			res.writeHead(400, { "Content-Type": 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
		}

		if (!model || !messages) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}

		try {
			const { instance, request } = await pool.requestChatCompletionInstance({
				model,
				messages,
			})

			const completion = await instance.processChatCompletion(request, (token) => {
				// TODO streaming
			})
			instance.unlock()

			res.writeHead(200, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify(completion, null, 2))

		} catch (e) {
			console.error(e)
			res.writeHead(500, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Internal server error' }))
		}
	}
}

// TODO https://platform.openai.com/docs/api-reference/completions/create
function createCompletionHandler(pool: LLMPool) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		res.writeHead(200, { 'Content-Type': 'application/json' })
		res.end(JSON.stringify({ foo: 'bar' }))
	}
}
// TODO https://platform.openai.com/docs/api-reference/models/list
function createListModelsHandler(pool: LLMPool) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		res.writeHead(200, { 'Content-Type': 'application/json' })
		res.end(JSON.stringify({ foo: 'bar' }))
	}
}

export function createOpenAIRequestHandlers(pool: LLMPool) {
	return {
		chatCompletions: createChatCompletionHandler(pool),
		completions: createCompletionHandler(pool),
		listModels: createListModelsHandler(pool),
	}
}
