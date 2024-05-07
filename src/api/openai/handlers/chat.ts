import { IncomingMessage, ServerResponse } from 'node:http'
import type { OpenAI } from 'openai'
import { LLMPool } from '#lllms/pool.js'
import { ChatCompletionRequest, ChatMessage } from '#lllms/types/index.js'
import { parseJSONRequestBody } from '#lllms/api/parseJSONRequestBody.js'
import { omitEmptyValues } from '#lllms/lib/util.js'
import { finishReasons } from '../finishReasons.js'

interface OpenAIChatCompletionParams
	extends Omit<OpenAI.ChatCompletionCreateParamsStreaming, 'stream'> {
	stream?: boolean
	top_k?: number
	min_p?: number
	repeat_penalty_num?: number
}

interface OpenAIChatCompletionChunk extends OpenAI.ChatCompletionChunk {
	usage?: OpenAI.CompletionUsage
}

// v1/chat/completions
// https://platform.openai.com/docs/api-reference/chat/create
export function createChatCompletionHandler(pool: LLMPool) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		let args: OpenAIChatCompletionParams

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
		req.on('aborted', () => {
			console.debug('Client aborted connection')
			controller.abort()
		})
		req.on('error', () => {
			console.debug('Client error')
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

			const completionReq = omitEmptyValues<ChatCompletionRequest>({
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
				tokenBias: args.logit_bias ? args.logit_bias : undefined,
				grammar: args.response_format?.type === 'json_object' ? 'json' : undefined,
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
			const completion = instance.createChatCompletion(completionReq)

			if (ssePing) {
				clearInterval(ssePing)
			}
			const result = await completion.process({
				signal: controller.signal,
				onChunk: (chunk) => {
					// console.debug('onChunk', chunk)
					if (args.stream) {
						const chunkData: OpenAIChatCompletionChunk = {
							id: completion.id,
							object: 'chat.completion.chunk',
							model: completion.model,
							created: Math.floor(completion.createdAt.getTime() / 1000),
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
			release()

			if (args.stream) {
				if (args.stream_options?.include_usage) {
					// beta chat completions pick up the meta data from the last chunk
					const finalChunk: OpenAIChatCompletionChunk = {
						id: completion.id,
						object: 'chat.completion.chunk',
						model: completion.model,
						created: Math.floor(completion.createdAt.getTime() / 1000),
						system_fingerprint: instance.fingerprint,
						choices: [
							{
								index: 0,
								delta: {},
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
					res.write(
						`data: ${JSON.stringify(finalChunk)}\n\n`,
					)
				}
				res.write('data: [DONE]')
				res.end()
			} else {
				const response: OpenAI.ChatCompletion = {
					id: completion.id,
					model: completion.model,
					object: 'chat.completion',
					created: Math.floor(completion.createdAt.getTime() / 1000),
					system_fingerprint: instance.fingerprint,
					choices: [
						{
							index: 0,
							message: {
								role: 'assistant',
								content: result.message.content,
							},
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
