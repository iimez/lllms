import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { Server } from 'node:http'
import OpenAI from 'openai'
import { createLLMServer, LLMServerOptions } from '../src/server.js'

const testConfig: LLMServerOptions = {
	concurrency: 1,
	models: {
		'orca-3b': {
			url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
			minInstances: 1,
			engine: 'gpt4all',
		},
		'phi3': {
			url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
			// minInstances: 1,
			engine: 'node-llama-cpp',
		}
	},
}

suite('OpenAI API Integration Tests', () => {
	let server: Server
	const openai = new OpenAI({
		baseURL: 'http://localhost:3000/v1/',
		apiKey: '123',
	})

	beforeAll(async () => {
		const res = await createLLMServer(testConfig)
		server = res.server
		server.listen(3000)
		await res.initPromise
	})

	afterAll(async () => {
		server.close()
	})

	test('Chat completion (gpt4all)', async () => {
		const completion = await openai.chat.completions.create({
			model: 'orca-3b',
			temperature: 0,
			messages: [
				{ role: 'user', content: 'This is a test. Just answer with "Test".' },
			],
		})
		expect(completion.choices[0].message.content).toContain('Test')
	})
	
	test('Chat completion stream (gpt4all)', async () => {
		const completion = await openai.chat.completions.create({
			model: 'orca-3b',
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
	
	test('Completion (gpt4all)', async () => {
		const completion = await openai.completions.create({
			model: 'orca-3b',
			temperature: 0,
			max_tokens: 1,
			prompt: 'To verify our code works we will first write an integration',
		})
		const firstWord = completion.choices[0].text.trim().split(' ')[0]
		expect(firstWord).toBe('test')
	})
	
	test('Completion streamed (gpt4all)', async () => {
		const completion = await openai.completions.create({
			model: 'orca-3b',
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
		expect(tokens.join('')).toContain('but some animals are more equal than others')
		expect(finishReason).toBe('stop')
	})
	
	

	test('Chat completion (node-llama-cpp)', async () => {
		const completion = await openai.chat.completions.create({
			model: 'phi3',
			temperature: 0,
			messages: [
				{ role: 'user', content: 'This is a test. Just answer with "Test".' },
			],
		})
		expect(completion.choices[0].message.content).toContain('Test')
	})
	
	test('Chat completion stream (node-llama-cpp)', async () => {
		const completion = await openai.chat.completions.create({
			model: 'phi3',
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
	
	test('Completion (node-llama-cpp)', async () => {
		const completion = await openai.completions.create({
			model: 'phi3',
			temperature: 0,
			max_tokens: 1,
			prompt: 'To verify our code works we will first write an integration',
		})
		const firstWord = completion.choices[0].text.trim().split(' ')[0]
		expect(firstWord).toBe('test')
	})
	
	test('Completion streamed (node-llama-cpp)', async () => {
		const completion = await openai.completions.create({
			model: 'phi3',
			temperature: 0,
			stream: true,
			stop: ['.'],
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
		expect(tokens.join('')).toContain('but some animals are more equal than others')
		expect(finishReason).toBe('stop')
	})
})
