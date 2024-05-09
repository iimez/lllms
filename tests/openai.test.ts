import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { Server } from 'node:http'
import OpenAI from 'openai'
import { serveLLMs } from '#lllms/http.js'

const testModel = 'test'

function runOpenAITests(client: OpenAI) {

	test('chat.completions.create', async () => {
		const completion = await client.chat.completions.create({
			model: testModel,
			temperature: 0,
			messages: [
				{ role: 'user', content: 'This is a test. Just answer with "Test".' },
			],
		})
		expect(completion.choices[0].message.content).toContain('Test')
	})

	test('chat.completions.create stream=true', async () => {
		const completion = await client.chat.completions.create({
			model: testModel,
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
			model: testModel,
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
		expect(finalResult.model).toBe(testModel)
		expect(finalResult.usage).toBeDefined()
		expect(finalResult.usage?.completion_tokens).toBeGreaterThan(0)
	})

	test('completions.create', async () => {
		const completion = await client.completions.create({
			model: testModel,
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
			model: testModel,
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
			model: testModel,
			temperature: 0,
			response_format: { type: 'json_object' },
			messages: [
				{ role: 'user', content: 'This is a test. Just answer with "Test", but in JSON. And add an array of cats to it.' },
			],
		})
		expect(completion.choices[0].message.content).toBeTruthy()
		const response = JSON.parse(completion.choices[0].message.content!)
		expect(response.result).toContain('Test')
		expect(response.cats).toBeInstanceOf(Array)
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
			log: 'debug',
			listen: { port: 3000 },
			inferenceConcurrency: 2,
			models: {
				[testModel]: {
					url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
					engine: 'node-llama-cpp',
					minInstances: 2,
				}
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
			log: 'debug',
			listen: { port: 3001 },
			inferenceConcurrency: 2,
			models: {
				[testModel]: {
					url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
					minInstances: 2,
					engine: 'gpt4all',
				}
			},
		})
	})

	afterAll(() => {
		server.close()
	})
	
	runOpenAITests(openai)
})
