import { suite, it, test, beforeAll, afterAll } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { LLMOptions } from '#lllms/types/index.js'
import {
	runStopTriggerTest,
	runTokenBiasTest,
	runSystemMessageTest,
	runContextLeakTest,
	runContextReuseTest,
	runContextShiftTest,
	runFunctionCallTest,
	runSequentialFunctionCallTest,
	runParallelFunctionCallTest,
	runGrammarTest,
} from './lib/index.js'

const models: Record<string, LLMOptions> = {
	test: {
		task: 'inference',
		// on llama3 instruct everything but parallel function calls works.
		// url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
		// sha256: '1977ae6185ef5bc476e27db85bb3d79ca4bd87e7b03399083c297d9c612d334c',
		// on functionary everything but the context shift test works.
		url: 'https://huggingface.co/meetkai/functionary-small-v2.5-GGUF/raw/main/functionary-small-v2.5.Q4_0.gguf',
		sha256: '3941bf2a5d1381779c60a7ccb39e8c34241e77f918d53c7c61601679b7160c48',
		engine: 'node-llama-cpp',
		contextSize: 2048,
	},
}

suite('Features', () => {
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

	test('token bias', async () => {
		await runTokenBiasTest(llms)
	})
	
	test('grammar', async () => {
		await runGrammarTest(llms)
	})
	
	test('function call', async () => {
		await runFunctionCallTest(llms)
	})
	
	test('sequential function calls', async () => {
		await runSequentialFunctionCallTest(llms)
	})
	
	test('parallel function calls', async () => {
		await runParallelFunctionCallTest(llms)
	})
})

suite('Context / Sessions', () => {
	const llms = new LLMServer({
		// log: 'debug',
		models: {
			test: {
				...models.test,
				maxInstances: 2,
			},
		},
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
	it('should do a context shift', async () => {
		await runContextShiftTest(llms)
	})
})
