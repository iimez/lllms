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
} from './lib/index.js'

const models: Record<string, LLMOptions> = {
	test: {
		// TODO has issues with templates.
		// url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
		// sha256: '8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef',
		// url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
		// sha256: '19ded996fe6c60254dc7544d782276eff41046ed42aa5f2d0005dc457e5c0895',
		url: 'https://huggingface.co/meetkai/functionary-small-v2.4-GGUF/resolve/main/functionary-small-v2.4.Q4_0.gguf',
		sha256: 'a092a87db6251a10a61ef4b91f30eb1d53c1660e0ddd1977832175ba589e2d58',
		engine: 'node-llama-cpp',
		contextSize: 2048,
		maxInstances: 2,
		functions: {
			getCurrentLocation: {
				description: 'Get the current location',
				handler: async () => {
					return 'New York, New York, United States'
				},
			},
		},
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

	// test('stop generation trigger', async () => {
	// 	await runStopTriggerTest(llms)
	// })

	// test('system message', async () => {
	// 	await runSystemMessageTest(llms)
	// })

	// test('token bias', async () => {
	// 	await runTokenBiasTest(llms)
	// })
	
	test('function calls', async () => {
		await runFunctionCallTest(llms)
	})
})

// suite('Context / Sessions', () => {
// 	const llms = new LLMServer({
// 		log: 'debug',
// 		models,
// 	})
// 	beforeAll(async () => {
// 		await llms.start()
// 	})
// 	afterAll(async () => {
// 		await llms.stop()
// 	})
// 	it('should reuse context on stateless requests', async () => {
// 		await runContextReuseTest(llms)
// 	})
// 	it('should not leak when handling multiple sessions', async () => {
// 		await runContextLeakTest(llms)
// 	})
// 	it('context shift', async () => {
// 		await runContextShiftTest(llms)
// 	})
// })
