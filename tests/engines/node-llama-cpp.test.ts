import { suite, it, test, beforeAll, afterAll } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { LLMOptions } from '#lllms/types/index.js'
import { createLogger } from '#lllms/lib/logger.js'
import {
	runContextLeakTest,
	runContextReuseTest,
	runStopParamTest,
	runSystemPromptTest,
} from './lib/index.js'

const models: Record<string, LLMOptions> = {
	test: {
		url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
		// sha256: '8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef',
		engine: 'node-llama-cpp',
		maxInstances: 2,
		templateFormat: 'phi',
	},
}

suite('Features', () => {
	const llms = new LLMServer({ models })

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	test('stop generation trigger', async () => {
		await runStopParamTest(llms)
	})

	test('system message', async () => {
		await runSystemPromptTest(llms)
	})
})

suite('Context / Sessions', () => {
	const llms = new LLMServer({
		// logger: createLogger('debug'),
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
