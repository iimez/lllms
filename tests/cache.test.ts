import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { LLMServer, LLMServerOptions } from '../src/server.js'
import { ChatMessage, ChatCompletionRequest } from '../src/types/index.js'

const testConfig: LLMServerOptions = {
	concurrency: 1,
	models: {
		// 'orca-3b': {
		// 	url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
		// 	minInstances: 1,
		// 	maxInstances: 2,
		// 	engine: 'gpt4all',
		// },
		'phi3-4k': {
			url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
			minInstances: 1,
			maxInstances: 2,
			engine: 'gpt4all',
		}
	},
}

async function createChatCompletion(
	server: LLMServer,
	args: ChatCompletionRequest,
) {
	const lock = await server.pool.requestCompletionInstance(args)
	const completion = lock.instance.createChatCompletion(args)
	const result = await completion.process()
	lock.release()
	return { completion, result }
}

function getInstanceId(completionId: string) {
	// part after the last ":" because model names may include colons
	const afterModelName = completionId.split(':').pop()
	// rest is instanceId-completionId
	const instanceId = afterModelName?.split('-')[0]
	return instanceId
}

suite('Cache Tests', () => {
	let llms: LLMServer

	beforeAll(async () => {
		llms = new LLMServer(testConfig)
		await llms.ready
	})

	afterAll(async () => {
		await llms.close()
	})

	test('Context reuse', async () => {
		// test whether the instance cache will be kept around for a follow up
		// while handling intermediate incoming completion requests.
		// middle part of the completion id is the instance uid. we can use this
		// to verify that the same or a different instance is used.
		const messagesA: ChatMessage[] = [
			{ role: 'user', content: 'Write a haiku about bears.' },
		]
		const responseA1 = await createChatCompletion(llms, {
			model: 'phi3-4k',
			temperature: 0,
			messages: messagesA,
		})
		const instanceIdA1 = getInstanceId(responseA1.completion.id)
		// do a unrelated chat completion that should be picked up by another instance
		const responseB1 = await createChatCompletion(llms, {
			model: 'phi3-4k',
			temperature: 0,
			messages: [
				{
					role: 'user',
					content:
						'Write a haiku about pancakes.',
				},
			],
		})
		const instanceIdB1 = getInstanceId(responseB1.completion.id)
		expect(instanceIdA1).not.toBe(instanceIdB1)
		// send a follow up turn to see if context is still there
		messagesA.push(responseA1.result.message, {
			role: 'user',
			content: 'Give it a 6 word title.',
		})
		const responseA2 = await createChatCompletion(llms, {
			model: 'phi3-4k',
			temperature: 0,
			messages: messagesA,
		})
		const instanceIdA2 = getInstanceId(responseA2.completion.id)
		expect(instanceIdA1).toBe(instanceIdA2)
	})

	test('Context reset', async () => {
		// test whether reused instances leak their context
		const messagesA: ChatMessage[] = [
			{
				role: 'user',
				content:
					"Please remember this fact for later: Axolotls can regenerate lost limbs. Don't forget! Just answer with 'OK'.",
			},
		]
		const responseA1 = await createChatCompletion(llms, {
			model: 'phi3-4k',
			temperature: 0,
			messages: messagesA,
			stop: ['OK'],
		})
		const instanceIdA1 = getInstanceId(responseA1.completion.id)
		const responseB1 = await createChatCompletion(llms, {
			model: 'phi3-4k',
			temperature: 0,
			stop: ['\n'],
			messages: [{ role: 'user', content: 'Remind me of one animal fact? One Sentence.' }],
		})
		const instanceIdB1 = getInstanceId(responseB1.completion.id)
		// expect response B1 to be handled by a different instance
		expect(instanceIdA1).not.toBe(instanceIdB1)
		// assert request B1 doesnt know the secret
		expect(responseB1.result.message.content).not.toMatch(/axolotl/i)

		messagesA.push(responseA1.result.message, {
			role: 'user',
			content: 'Remind me of one animal fact? One Sentence.',
		})
		const responseA2 = await createChatCompletion(llms, {
			model: 'phi3-4k',
			temperature: 0,
			stop: ['\n'],
			messages: messagesA,
		})
		const instanceIdA2 = getInstanceId(responseA2.completion.id)
		// assert follow up turn is handled by the same instance
		expect(instanceIdA1).toBe(instanceIdA2)
		// assert request A2 does know the secret
		expect(responseA2.result.message.content).toMatch(/axolotl/i)
	})
})
