import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { Server } from 'node:http'
import OpenAI from 'openai'
import { serveLLMs } from '#lllms/http.js'

const chatModel = 'chat'
const embeddingsModel = 'text-embed'

function runOpenAITests(client: OpenAI) {
	test('chat.completions.create', async () => {
		const completion = await client.chat.completions.create({
			model: chatModel,
			temperature: 0,
			messages: [
				{ role: 'user', content: 'This is a test. Just answer with "Test".' },
			],
		})
		expect(completion.choices[0].message.content).toContain('Test')
	})

	test('chat.completions.create stream=true', async () => {
		const completion = await client.chat.completions.create({
			model: chatModel,
			temperature: 0,
			stream: true,
			messages: [
				{ role: 'user', content: 'This is a test. Just answer with "Test".' },
			],
		})
		let receivedTestChunk = false
		for await (const chunk of completion) {
			if (chunk.choices[0]?.delta?.content?.includes('Test')) {
				receivedTestChunk = true
			}
		}
		expect(receivedTestChunk).toBe(true)
	})

	test('beta.chat.completions.create stream=true', async () => {
		const completion = await client.beta.chat.completions.stream({
			stream_options: { include_usage: true },
			model: chatModel,
			temperature: 0,
			messages: [
				{ role: 'user', content: 'This is a test. Just answer with "Test".' },
			],
		})
		let receivedTestChunk = false
		for await (const chunk of completion) {
			if (chunk.choices[0]?.delta?.content?.includes('Test')) {
				receivedTestChunk = true
			}
		}
		const finalResult = await completion.finalChatCompletion()
		expect(receivedTestChunk).toBe(true)
		expect(finalResult.model).toBe(chatModel)
		expect(finalResult.usage).toBeDefined()
		expect(finalResult.usage?.completion_tokens).toBeGreaterThan(0)
	})

	test('completions.create', async () => {
		const completion = await client.completions.create({
			model: chatModel,
			temperature: 0,
			max_tokens: 1,
			prompt: 'To verify our code works we will first write an integration',
		})
		const firstWord = completion.choices[0].text.trim().split(' ')[0]
		expect(firstWord).toBe('test')
	})

	test('completions.create stream=true', async () => {
		const completion = await client.completions.create({
			stream_options: { include_usage: true },
			model: chatModel,
			temperature: 0,
			stream: true,
			stop: ['.'],
			max_tokens: 100,
			prompt: '"All animals are equal,',
		})
		const tokens: string[] = []
		let finishReason: string = ''
		for await (const chunk of completion) {
			if (chunk.choices[0].text) {
				tokens.push(chunk.choices[0].text)
			}
			finishReason = chunk.choices[0].finish_reason
		}
		console.debug({ response: tokens.join('|'), finishReason })
		expect(tokens.join('')).toContain(
			'but some animals are more equal than others',
		)
		expect(finishReason).toBe('stop')
	})

	test('response_format json_object / json grammar', async () => {
		const completion = await client.chat.completions.create({
			model: chatModel,
			temperature: 0,
			response_format: { type: 'json_object' },
			messages: [
				{
					role: 'user',
					content:
						'Answer with a JSON object containing the key "test" with the value "test". And add an array of cats to it.',
				},
			],
		})
		expect(completion.choices[0].message.content).toBeTruthy()
		const response = JSON.parse(completion.choices[0].message.content!)
		expect(response.test).toMatch(/test/)
		expect(response.cats).toBeInstanceOf(Array)
	})

	test('function call', async () => {
		const completion = await client.chat.completions.create({
			model: chatModel,
			temperature: 0,
			tools: [
				{
					type: 'function',
					function: {
						name: 'getRandomNumber',
						description: 'Generate a random integer in given range',
						parameters: {
							type: 'object',
							properties: {
								min: { type: 'number' },
								max: { type: 'number' },
							},
						},
					},
				},
			],
			messages: [{ role: 'user', content: "Let's roll the die." }],
		})
		expect(completion.choices[0].message.content).toBeNull()
		expect(completion.choices[0].message.tool_calls).toBeInstanceOf(Array)
		expect(completion.choices[0].message.tool_calls![0].type).toBe('function')
		expect(completion.choices[0].message.tool_calls![0].function.name).toBe(
			'getRandomNumber',
		)
		expect(
			completion.choices[0].message.tool_calls![0].function.arguments,
		).toBe('{"min":1,"max":6}')
	})

	test('function call with streaming', async () => {
		const completion = await client.beta.chat.completions.stream({
			stream_options: { include_usage: true },
			model: chatModel,
			temperature: 0,
			tools: [
				{
					type: 'function',
					function: {
						name: 'getRandomNumber',
						description: 'Generate a random integer in given range',
						parameters: {
							type: 'object',
							properties: {
								min: { type: 'number' },
								max: { type: 'number' },
							},
						},
					},
				},
			],
			messages: [{ role: 'user', content: "Let's roll the die." }],
		})
		let receivedToolCallChunk = false
		for await (const chunk of completion) {
			if (chunk.choices[0]?.delta?.tool_calls?.length) {
				receivedToolCallChunk = true
			}
		}
		expect(receivedToolCallChunk).toBe(true)
		
		const finalResult = await completion.finalChatCompletion()
		expect(finalResult.choices[0].message.tool_calls).toBeInstanceOf(Array)
		expect(finalResult.choices[0].message.tool_calls![0].type).toBe('function')
		expect(finalResult.choices[0].message.tool_calls![0].function.name).toBe('getRandomNumber')
		expect(finalResult.choices[0].message.tool_calls![0].function.arguments).toBe('{"min":1,"max":6}')
		expect(finalResult.model).toBe(chatModel)
		expect(finalResult.usage).toBeDefined()
		expect(finalResult.usage?.completion_tokens).toBeGreaterThan(0)
	})
	
	test('embeddings.create', async () => {
		const res = await client.embeddings.create({
			model: embeddingsModel,
			input: 'This is a test.',
		})
		expect(res.data).toBeInstanceOf(Array)
		expect(res.data[0].embedding).toBeInstanceOf(Array)
		expect(Number.isFinite(res.data[0].embedding[0])).toBe(true)
	})
}

suite('OpenAI API (node-llama-cpp)', () => {
	let server: Server
	const openai = new OpenAI({
		baseURL: 'http://localhost:3000/openai/v1/',
		apiKey: '123',
	})

	beforeAll(async () => {
		server = await serveLLMs({
			// log: 'debug',
			listen: { port: 3000 },
			concurrency: 2,
			models: {
				[embeddingsModel]: {
					url: 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/blob/main/nomic-embed-text-v1.5.Q8_0.gguf',
					sha256: '3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7',
					minInstances: 1,
					engine: 'node-llama-cpp',
					task: 'embedding',
					engineOptions: {
						gpu: false,
					}
				},
				[chatModel]: {
					// url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf',
					// sha256: 'c57380038ea85d8bec586ec2af9c91abc2f2b332d41d6cf180581d7bdffb93c1',
					url: 'https://huggingface.co/meetkai/functionary-small-v2.5-GGUF/raw/main/functionary-small-v2.5.Q4_0.gguf',
					sha256: '3941bf2a5d1381779c60a7ccb39e8c34241e77f918d53c7c61601679b7160c48',
					engine: 'node-llama-cpp',
					minInstances: 1,
					task: 'inference',
					// engineOptions: {
					// 	gpu: 'vulkan',
					// }
				},
			},
		})
	})

	afterAll(() => {
		server.close()
	})

	runOpenAITests(openai)
})

suite('OpenAI API (gpt4all)', () => {
	let server: Server
	const openai = new OpenAI({
		baseURL: 'http://localhost:3001/openai/v1/',
		apiKey: '123',
	})

	beforeAll(async () => {
		server = await serveLLMs({
			// log: 'debug',
			listen: { port: 3001 },
			concurrency: 2,
			models: {
				[embeddingsModel]: {
					url: 'https://gpt4all.io/models/gguf/nomic-embed-text-v1.f16.gguf',
					minInstances: 1,
					engine: 'gpt4all',
					task: 'embedding',
				},
				[chatModel]: {
					url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
					minInstances: 1,
					engine: 'gpt4all',
					task: 'inference',
				}
			},
		})
	})

	afterAll(() => {
		server.close()
	})

	runOpenAITests(openai)
})
