import { suite, it, test, beforeAll, afterAll } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { LLMOptions } from '#lllms/types/index.js'
import {
	runContextLeakTest,
	runContextReuseTest,
	runStopTriggerTest,
	runSystemMessageTest,
} from './lib/index.js'

const models: Record<string, LLMOptions> = {
	test: {
		url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
		// md5: 'f8347badde9bfc2efbe89124d78ddaf5',
		engine: 'gpt4all',
		maxInstances: 2,
	},
}

suite('Features', () => {
	const llms = new LLMServer({
		log: 'debug',
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

suite('Context / Sessions', () => {
	const llms = new LLMServer({ models })

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
