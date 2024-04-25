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

const finishReasonMap = {
	maxTokens: 'length',
	functionCall: 'function_call',
	eosToken: 'stop',
	stopGenerationTrigger: 'stop',
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
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
		}

		if (!model || !messages) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}
		
		req.on('close', () => {
			// TODO abort generation if client closes connection
			console.log('Client closed connection')
		})

		try {

			const { instance, release } = await pool.requestChatCompletionInstance({
				model,
				messages,
			})
			const completion = await instance.processChatCompletion({
				model,
				messages,
			})
			release()
			
			const response = {
				id: completion.id,
				model: completion.model,
				object: 'chat.completion',
				created: Math.floor(Date.now() / 1000),
				system_fingerprint: instance.fingerprint,
				choices: [
					{
						index: 0,
						message: completion.message,
						logprobs: null,
						finish_reason: finishReasonMap[completion.finishReason],
					},
				],
				usage: {
					prompt_tokens: completion.promptTokens,
					completion_tokens: completion.completionTokens,
					total_tokens: completion.totalTokens,
				},
			}
			res.writeHead(200, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify(response, null, 2))
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
	    res.end(JSON.stringify({
                object: 'list',
                data: Object
                        .values(pool.instances)
                        .map(v => ({ object: "model",
                                     owned_by: v.config.engine,
                                     id: v.config.name,
                                     //https://stackoverflow.com/a/51442878
                                     created: Math.floor(v.createdBy.getTime() / 1000)  })) }))
	}
}

export function createOpenAIRequestHandlers(pool: LLMPool) {
	return {
		chatCompletions: createChatCompletionHandler(pool),
		completions: createCompletionHandler(pool),
		listModels: createListModelsHandler(pool),
	}
}
