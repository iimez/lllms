import { IncomingMessage, ServerResponse } from 'node:http'
import type { OpenAI } from 'openai'
import { LLMPool } from '#lllms/pool.js'
import { CompletionRequest } from '#lllms/types/index.js'
import { parseJSONRequestBody } from '#lllms/api/parseJSONRequestBody.js'
import { omitEmptyValues } from '#lllms/lib/util.js'
import { finishReasons } from '../finishReasons.js'

interface OpenAICompletionParams
	extends Omit<OpenAI.CompletionCreateParamsStreaming, 'stream'> {
	stream?: boolean
	top_k?: number
	min_p?: number
	repeat_penalty_num?: number
}

interface OpenAICompletionChunk extends OpenAI.Completions.Completion {
	usage?: OpenAI.CompletionUsage
}

// v1/completions
// https://platform.openai.com/docs/api-reference/completions/create
export function createCompletionHandler(pool: LLMPool) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		let args: OpenAICompletionParams

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
		if (!args.model || !args.prompt) {
			res.writeHead(400, { 'Content-Type': 'application/json' })
			res.end(JSON.stringify({ error: 'Invalid request' }))
			return
		}
		if (!pool.config.models[args.model]) {
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

			const completionReq = omitEmptyValues<CompletionRequest>({
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
				tokenBias: args.logit_bias ? args.logit_bias : undefined,
				topP: args.top_p ? args.top_p : undefined,
				// additional non-spec params
				repeatPenaltyNum: args.repeat_penalty_num
					? args.repeat_penalty_num
					: undefined,
				minP: args.min_p ? args.min_p : undefined,
				topK: args.top_k ? args.top_k : undefined,
			})

			const { instance, release } = await pool.requestLLM(
				completionReq,
				controller.signal,
			)
			const completion = instance.createCompletion(completionReq)

			const result = await completion.process({
				signal: controller.signal,
				onChunk: (chunk) => {
					if (args.stream) {
						const chunkData: OpenAICompletionChunk = {
							id: completion.id,
							model: completion.model,
							object: 'text_completion',
							created: Math.floor(completion.createdAt.getTime() / 1000),
							choices: [
								{
									index: 0,
									text: chunk.text,
									logprobs: null,
									// @ts-ignore official api returns null here in the same case
									finish_reason: null,
								},
							],
						}
						res.write(`data: ${JSON.stringify(chunkData)}\n\n`)
					}
				},
			})
			release()

			if (args.stream) {
				if (args.stream_options?.include_usage) {
					const finalChunk: OpenAICompletionChunk = {
						id: completion.id,
						model: completion.model,
						object: 'text_completion',
						created: Math.floor(completion.createdAt.getTime() / 1000),
						choices: [
							{
								index: 0,
								text: '',
								logprobs: null,
								// @ts-ignore
								finish_reason: finishReasons[result.finishReason],
							},
						],
					}
					res.write(
						`data: ${JSON.stringify(finalChunk)}\n\n`,
					)
				}
				res.write('data: [DONE]')
				res.end()
			} else {
				const response: OpenAI.Completions.Completion = {
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
							// @ts-ignore
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
