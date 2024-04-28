import { IncomingMessage, ServerResponse } from 'node:http'
import type { Request } from 'express'
import { existsSync, statSync } from 'node:fs'
import path from 'node:path'
import type { OpenAI } from 'openai'
import { LLMPool } from '../pool.js'
import { ChatMessage, CompletionFinishReason } from '../types/index.js'

function parseJSONRequestBody(req: IncomingMessage | Request): Promise<any> {
	return new Promise((resolve, reject) => {

		// if request is from express theres no need to parse anything
		if ('body' in req) {
			resolve(req.body)
			return
		}
		
		// for native http server
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

const openAiFinishReasonMap: Record<CompletionFinishReason, string> = {
	maxTokens: 'length',
	functionCall: 'function_call', // TODO tool_calls
	eogToken: 'stop',
	stopGenerationTrigger: 'stop',
} as const

// See OpenAI API specs at https://github.com/openai/openai-openapi/blob/master/openapi.yaml

// v1/chat/completions
// https://platform.openai.com/docs/api-reference/chat/create
function createChatCompletionHandler(pool: LLMPool) {
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

		const controller = new AbortController()
		req.on('close', () => {
			console.debug('Client closed connection')
			controller.abort()
		})

		try {
			if (args.stream) {
				res.writeHead(200, {
					'Content-Type': 'text/event-stream',
					'Cache-Control': 'no-cache',
					'Connection': 'keep-alive',
				})
				res.flushHeaders()
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
			const { instance, releaseInstance } = await pool.requestCompletionInstance(
				completionReq,
				controller.signal,
			)
			const completion = instance.createChatCompletion(completionReq)

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
										content: chunk.token,
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
									? openAiFinishReasonMap[result.finishReason]
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
								? openAiFinishReasonMap[result.finishReason]
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

// v1/completions
// https://platform.openai.com/docs/api-reference/completions/create
function createCompletionHandler(pool: LLMPool) {
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

			const { instance, releaseInstance } = await pool.requestCompletionInstance(
				completionReq,
				controller.signal,
			)
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
									text: chunk.token,
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
									? openAiFinishReasonMap[result.finishReason]
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
								? openAiFinishReasonMap[result.finishReason]
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

// https://platform.openai.com/docs/api-reference/models/list
function createListModelsHandler(pool: LLMPool) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		const data = Object.entries(pool.config.models).map(([id, config]) => {
			let created = 0
			if (existsSync(config.file)) {
				//https://stackoverflow.com/a/51442878
				created = Math.floor(statSync(config.file).birthtime.getTime() / 1000)
			}
			// TODO possible to get created/owned_by from gguf metadata?
			return {
				object: 'model',
				owned_by: path.basename(config.file),
				id,
				created,
			}
		})

		res.writeHead(200, { 'Content-Type': 'application/json' })
		res.end(JSON.stringify({ object: 'list', data }, null, 2))
	}
}

export function createOpenAIRequestHandlers(pool: LLMPool) {
	return {
		chatCompletions: createChatCompletionHandler(pool),
		completions: createCompletionHandler(pool),
		listModels: createListModelsHandler(pool),
	}
}
