import { IncomingMessage, ServerResponse } from 'node:http'
import type { OpenAI } from 'openai'
import { LLMPool } from '../../../pool.js'
import { parseJSONRequestBody } from '../parseJSONRequestBody.js'
import { finishReasons } from '../finishReasons.js'

// v1/completions
// https://platform.openai.com/docs/api-reference/completions/create
export function createCompletionHandler(pool: LLMPool) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		let args:
			| OpenAI.CompletionCreateParamsStreaming
			| OpenAI.CompletionCreateParams

		try {
			const body = await parseJSONRequestBody(req)
			args = body
		} catch (e) {
			console.error(e)
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}

		if (!args.model || !args.prompt) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
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
			if (args.stream) {
				res.writeHead(200, {
					'Content-Type': 'text/event-stream',
					'Cache-Control': 'no-cache',
					Connection: 'keep-alive',
				})
				res.flushHeaders()
			}

			let prompt = args.prompt

			if (typeof prompt !== 'string') {
				throw new Error('Prompt must be a string')
			}

			let stop = args.stop ? args.stop : undefined
			if (typeof stop === 'string') {
				stop = [stop]
			}

			const completionReq = {
				model: args.model,
				prompt: args.prompt as string,
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
			const completion = instance.createCompletion(completionReq)

			const result = await completion.process({
				signal: controller.signal,
				onChunk: (chunk) => {
					if (args.stream) {
						const chunkData = {
							id: completion.id,
							object: 'text_completion.chunk',
							choices: [
								{
									index: 0,
									text: chunk.text,
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
				res.write(
					`data: ${JSON.stringify({
						id: completion.id,
						object: 'text_completion.chunk',
						choices: [
							{
								index: 0,
								text: '',
								logprobs: null,
								finish_reason: result.finishReason
									? finishReasons[result.finishReason]
									: '',
							},
						],
					})}\n\n`,
				)
				res.write('data: [DONE]')
				res.end()
			} else {
				const response = {
					id: completion.id,
					model: completion.model,
					object: 'text_completion',
					created: Math.floor(completion.createdAt.getTime() / 1000),
					system_fingerprint: instance.fingerprint,
					choices: [
						{
							index: 0,
							text: result.text,
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
		} catch (err) {
			console.error(err)
			if (args.stream) {
				res.write('data: [ERROR]')
			} else {
				res.writeHead(500, { 'Content-Type': 'application/json' })
				res.end(JSON.stringify({ error: 'Internal server error' }))
			}
		}
	}
}
