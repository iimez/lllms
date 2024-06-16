import { IncomingMessage, ServerResponse } from 'node:http'
import type { OpenAI } from 'openai'
import { EmbeddingsRequest } from '#lllms/types/index.js'
import { parseJSONRequestBody } from '#lllms/api/parseJSONRequestBody.js'
import { omitEmptyValues } from '#lllms/lib/util.js'
import { LLMServer } from '#lllms/server.js'

type OpenAIEmbeddingsParams = OpenAI.EmbeddingCreateParams

// v1/embeddings
// https://platform.openai.com/docs/api-reference/embeddings
export function createEmbeddingsHandler(llms: LLMServer) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		let args: OpenAIEmbeddingsParams

		try {
			const body = await parseJSONRequestBody(req)
			args = body
		} catch (e) {
			console.error(e)
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}
		
		// TODO ajv schema validation?
		if (!args.model || !args.input) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}
		if (!llms.getModelConfig(args.model)) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid model' }))
			return
		}

		const controller = new AbortController()
		req.on('close', () => {
			console.debug('Client closed connection')
			controller.abort()
		})
		req.on('end', () => {
			console.debug('Client ended connection')
			controller.abort()
		})

		try {


			let input = args.input

			if (typeof input !== 'string') {
				throw new Error('Input must be a string')
			}

			const embeddingsReq = omitEmptyValues<EmbeddingsRequest>({
				model: args.model,
				input: args.input as string,
			})

			const { instance, release } = await llms.requestInstance(
				embeddingsReq,
				controller.signal,
			)
			const result = await instance.createEmbeddings(embeddingsReq)

			release()

			const response: OpenAI.CreateEmbeddingResponse = {
				model: instance.model,
				object: 'list',
				data: result.embeddings.map((embedding, index) => ({
					embedding: Array.from(embedding),
					index,
					object: 'embedding',
				})),
				usage: {
					prompt_tokens: result.inputTokens,
					total_tokens: result.inputTokens,
				},
			}
			res.writeHead(200, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify(response, null, 2))

		} catch (err) {
			console.error(err)
			res.writeHead(500, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Internal server error' }))
		}
	}
}
