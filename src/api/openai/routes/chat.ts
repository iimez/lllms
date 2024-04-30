import { IncomingMessage, ServerResponse } from 'node:http'
import type { OpenAI } from 'openai'
import { LLMPool } from '../../../pool.js'
import { ChatMessage } from '../../../types/index.js'
import { parseJSONRequestBody } from '../parseJSONRequestBody.js'
import { finishReasons } from '../finishReasons.js'

// v1/chat/completions
// https://platform.openai.com/docs/api-reference/chat/create
export function createChatCompletionHandler(pool: LLMPool) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		let args:
			| OpenAI.ChatCompletionCreateParamsStreaming
			| OpenAI.ChatCompletionCreateParams

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
		if (!args.model || !args.messages) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}

		if (!pool.modelExists(args.model)) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid model' }))
			return
		}

		console.debug('Chat Completion', JSON.stringify(args, null, 2))

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
			let ssePing: NodeJS.Timeout | undefined
			if (args.stream) {
				res.writeHead(200, {
					'Content-Type': 'text/event-stream',
					'Cache-Control': 'no-cache',
					Connection: 'keep-alive',
				})
				res.flushHeaders()
				ssePing = setInterval(() => {
					res.write(':ping\n\n')
				}, 30000)
			}

			let stop = args.stop ? args.stop : undefined
			if (typeof stop === 'string') {
				stop = [stop]
			}

			const completionReq = {
				model: args.model,
				// TODO support multimodal image content array
				messages: args.messages.filter((m) => {
					return typeof m.content === 'string'
				}) as ChatMessage[],
				temperature: args.temperature ? args.temperature : undefined,
				stream: args.stream ? Boolean(args.stream) : false,
				maxTokens: args.max_tokens ? args.max_tokens : undefined,
				seed: args.seed ? args.seed : undefined,
				stop,
				frequencyPenalty: args.frequency_penalty
					? args.frequency_penalty
					: undefined,
				presencePenalty: args.presence_penalty
					? args.presence_penalty
					: undefined,
				topP: args.top_p ? args.top_p : undefined,
			}
			const { instance, releaseInstance } =
				await pool.requestCompletionInstance(completionReq, controller.signal)
			const completion = instance.createChatCompletion(completionReq)

			if (ssePing) {
				clearInterval(ssePing)
			}
			const result = await completion.process({
				signal: controller.signal,
				onChunk: (chunk) => {
					// console.debug('onChunk', chunk)
					if (args.stream) {
						const chunkData = {
							id: completion.id,
							object: 'chat.completion.chunk',
							choices: [
								{
									index: 0,
									delta: {
										role: 'assistant',
										content: chunk.text,
									},
									logprobs: null,
									finish_reason: null,
								},
							],
						}
						res.write(`data: ${JSON.stringify(chunkData)}\n\n`)
					}
				},
			})
			releaseInstance()

			if (args.stream) {
				// beta chat completions pick up the meta data from the last chunk
				res.write(
					`data: ${JSON.stringify({
						id: completion.id,
						model: completion.model,
						object: 'chat.completion.chunk',
						created: Math.floor(completion.createdAt.getTime() / 1000),
						system_fingerprint: instance.fingerprint,
						choices: [
							{
								index: 0,
								delta: {},
								logprobs: null,
								finish_reason: result.finishReason
									? finishReasons[result.finishReason]
									: '',
							},
						],
						usage: {
							prompt_tokens: result.promptTokens,
							completion_tokens: result.completionTokens,
							total_tokens: result.totalTokens,
						},
					})}\n\n`,
				)
				res.write('data: [DONE]')
				res.end()
			} else {
				const response = {
					id: completion.id,
					model: completion.model,
					object: 'chat.completion',
					created: Math.floor(completion.createdAt.getTime() / 1000),
					system_fingerprint: instance.fingerprint,
					choices: [
						{
							index: 0,
							message: result.message,
							logprobs: null,
							finish_reason: result.finishReason
								? finishReasons[result.finishReason]
								: 'stop',
						},
					],
					usage: {
						prompt_tokens: result.promptTokens,
						completion_tokens: result.completionTokens,
						total_tokens: result.totalTokens,
					},
				}
				res.writeHead(200, { 'Content-Type': 'application/json' })
				res.end(JSON.stringify(response, null, 2))
			}
		} catch (e) {
			console.error(e)
			if (args.stream) {
				res.write('data: [ERROR]')
			} else {
				res.writeHead(500, { 'Content-Type': 'application/json' })
				res.end(JSON.stringify({ error: 'Internal server error' }))
			}
		}
	}
}
