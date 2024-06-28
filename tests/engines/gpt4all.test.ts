import { suite, it, test, beforeAll, afterAll, expect } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { ChatCompletionRequest, ChatMessage, LLMOptions } from '#lllms/types/index.js'
import {
	runContextLeakTest,
	runContextReuseTest,
	runStopTriggerTest,
	runSystemMessageTest,
} from './lib/index.js'
import { createChatCompletion } from '../util.js'

const models: Record<string, LLMOptions> = {
	test: {
		task: 'text-completion',
		url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
		md5: 'c87ad09e1e4c8f9c35a5fcef52b6f1c9',
		// url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
		// md5: 'f8347badde9bfc2efbe89124d78ddaf5',
		engine: 'gpt4all',
		maxInstances: 2,
	},
}

suite('features', () => {
	const llms = new LLMServer({
		// log: 'debug',
		models,
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('stop generation trigger', async () => {
		await runStopTriggerTest(llms)
	})

	test('system message', async () => {
		await runSystemMessageTest(llms)
	})
})

suite('cache', () => {
	const llms = new LLMServer({
		// log: 'debug',
		models,
	})

	beforeAll(async () => {
		await llms.start()
	})

	afterAll(async () => {
		await llms.stop()
	})

	it('should reuse context on stateless requests', async () => {
		await runContextReuseTest(llms)
	})

	it('should not leak when handling multiple sessions', async () => {
		await runContextLeakTest(llms)
	})
})

suite('preload', () => {
	const preloadedMessages: ChatMessage[] = [
		{
			role: 'system',
			content: 'You are an advanced mathematician.',
		},
		{
			role: 'user',
			content: 'Whats 2+2?',
		},
		{
			role: 'assistant',
			content: "It's 5!",
		},
	]
	const llms = new LLMServer({
		models: {
			test: {
				task: 'text-completion',
				url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
				md5: 'c87ad09e1e4c8f9c35a5fcef52b6f1c9',
				engine: 'gpt4all',
				prepare: 'blocking',
				maxInstances: 2,
				preload: {
					messages: preloadedMessages,
				},
			},
		},
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('should utilize preloaded messages', async () => {
		const args: ChatCompletionRequest = {
			model: 'test',
			messages: [
				...preloadedMessages,
				{
					role: 'user',
					content: 'Are you sure?',
				},
			],
		}
		
		const lock = await llms.pool.requestInstance(args)
		// @ts-ignore
		const internalMessages = lock.instance.llm.activeChatSession.messages
		expect(internalMessages.length).toBe(2)
		await lock.release()
	})

	test('should not utilize preloaded messages', async () => {
		const args: ChatCompletionRequest = {
			model: 'test',
			messages: [
				{
					role: 'user',
					content: 'Whats 2+2?',
				},
			],
		}
		
		const lock = await llms.pool.requestInstance(args)
		// @ts-ignore
		// const internalMessagesBefore = lock.instance.llm.activeChatSession.messages
		// console.debug({
		// 	internalMessagesBefore,
		// })
		const handle = lock.instance.createChatCompletion(args)
		await handle.process()
		await lock.release()
		// @ts-ignore
		const internalMessagesAfter = lock.instance.llm.activeChatSession.messages
		// console.debug({
		// 	internalMessagesAfter,
		// })
		expect(internalMessagesAfter[1].content).not.toBe('It\'s 5!')
	})
})