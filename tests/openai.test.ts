import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { Server } from 'node:http'
import OpenAI from 'openai'
import { serveLLMs } from '../src/server.js'

const testModel = 'phi3-mini-4k'

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
			model: testModel,
			temperature: 0,
			stream: true,
			stop: ['."', '.'],
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
		expect(tokens.join('')).toContain(
			'but some animals are more equal than others',
		)
		expect(finishReason).toBe('stop')
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
			listen: { port: 3000 },
			concurrency: 2,
			models: {
				[testModel]: {
					url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
					engine: 'node-llama-cpp',
					minInstances: 2,
					templateFormat: 'phi',
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
			listen: { port: 3001 },
			concurrency: 2,
			models: {
				[testModel]: {
					url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
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
