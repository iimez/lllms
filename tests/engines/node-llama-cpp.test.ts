import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import fs from 'node:fs'
import { ModelServer } from '#lllms/server.js'
import {
	ChatCompletionRequest,
	ChatMessage,
	ModelOptions,
} from '#lllms/types/index.js'
import { ContextShiftStrategy } from '#lllms/engines/node-llama-cpp/types.js'
import {
	runStopTriggerTest,
	runTokenBiasTest,
	runSystemMessageTest,
	runContextLeakTest,
	runContextReuseTest,
	runFileIngestionTest,
	runGenerationContextShiftTest,
	runIngestionContextShiftTest,
	runFunctionCallTest,
	runSequentialFunctionCallTest,
	runParallelFunctionCallTest,
	runBuiltInGrammarTest,
	runRawGBNFGrammarTest,
	runJsonSchemaGrammarTest,
	runTimeoutTest,
	runCancellationTest,
} from './lib/index.js'
import { createChatCompletion } from '../util.js'

const testModel: ModelOptions = {
	url: 'https://huggingface.co/mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf',
	sha256: '56e1a31ac6e5037174344ac2153c33d873f301f2a312ef2619775190aade51c7',
	engine: 'node-llama-cpp',
	task: 'text-completion',
	contextSize: 2048,
	prepare: 'blocking',
	grammars: {
		'custom-gbnf-string': fs.readFileSync(
			'tests/fixtures/grammar/name-age-json.gbnf',
			'utf-8',
		),
		'custom-json-schema': {
			type: 'object',
			properties: {
				name: {
					type: 'string',
				},
				age: {
					type: 'number',
				},
			},
			required: ['name', 'age'],
		},
	},
	device: {
		gpu: 'vulkan',
	},
}

suite('features', () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
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

	test('token bias', async () => {
		await runTokenBiasTest(llms)
	})
})

suite('function calling', async () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: {
				task: 'text-completion',
				url: 'https://huggingface.co/meetkai/functionary-small-v3.2-GGUF/blob/main/functionary-small-v3.2.Q4_0.gguf',
				sha256: 'c0afdbbffa498a8490dea3401e34034ac0f2c6e337646513a7dbc04fcef1c3a4',
				engine: 'node-llama-cpp',
			},
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('basic function call', async () => {
		await runFunctionCallTest(llms)
	})
	test('sequential function calls', async () => {
		await runSequentialFunctionCallTest(llms)
	})
	test('parallel function calls', async () => {
		await runParallelFunctionCallTest(llms)
	})
})

suite('grammar', async () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('built-in grammar', async () => {
		await runBuiltInGrammarTest(llms)
	})

	test('gbnf string grammar', async () => {
		await runRawGBNFGrammarTest(llms)
	})

	test('json schema grammar', async () => {
		await runJsonSchemaGrammarTest(llms)
	})
})

suite('cache', () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: {
				...testModel,
				maxInstances: 2,
				device: { gpu: 'auto' },
			},
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('reuse existing instance on stateless requests', async () => {
		await runContextReuseTest(llms)
	})
	test('no leak when handling multiple sessions', async () => {
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
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: {
				...testModel,
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
		const chat = await createChatCompletion(llms, {
			model: 'test',
			messages: [
				...preloadedMessages,
				{
					role: 'user',
					content: 'Are you sure?',
				},
			],
		})
		expect(chat.result.contextTokens).toBeGreaterThan(80)
	})

	test('should not utilize preloaded messages', async () => {
		const chat = await createChatCompletion(llms, {
			model: 'test',
			messages: [
				{
					role: 'system',
					content: 'You are an advanced mathematician.',
				},
				{
					role: 'user',
					content: 'Whats 2+2?',
				},
			],
		})
		expect(chat.result.contextTokens).toBe(chat.result.promptTokens + chat.result.completionTokens)
	})
})

suite('context shift', () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('during first user message', async () => {
		await runIngestionContextShiftTest(llms)
	})
	test('during assistant response', async () => {
		await runGenerationContextShiftTest(llms)
	})
})

suite('ingest', () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: testModel,
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('normal text', async () => {
		const res = await runFileIngestionTest(llms, 'lovecraft')
		expect(res.message.content).toMatch(/horror|lovecraft/i)
	})
	test('a small website', async () => {
		const res = await runFileIngestionTest(llms, 'hackernews')
		expect(res.message.content).toMatch(/hacker|news/i)
	})
	test('a large website', async () => {
		const res = await runFileIngestionTest(llms, 'github')
		expect(res.message.content).toMatch(/github|html/i)
	})
})

suite('timeout and cancellation', () => {
	const llms = new ModelServer({
		// log: 'debug',
		models: {
			test: {
				...testModel,
				minInstances: 1,
				device: { gpu: true },
			},
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('timeout', async () => {
		await runTimeoutTest(llms)
	})
	test('cancellation', async () => {
		await runCancellationTest(llms)
	})
})
