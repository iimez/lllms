import { suite, test, expect, beforeAll, afterAll } from 'vitest'
import { LLMServer } from '../src/server.js'
import { ChatMessage, ChatCompletionRequest } from '../src/types/index.js'

const testModel = 'phi3-mini-4k'

function getInstanceId(completionId: string) {
	// part after the last ":" because model names may include colons
	const afterModelName = completionId.split(':').pop()
	const instanceId = afterModelName?.split('-')[0] // rest is instanceId-completionId
	return instanceId
}

async function createChatCompletion(
	server: LLMServer,
	args: ChatCompletionRequest,
) {
	const lock = await server.pool.requestCompletionInstance(args)
	const completion = lock.instance.createChatCompletion(args)
	const result = await completion.process()
	lock.releaseInstance()
	return { completion, result }
}

// conversation that tests whether the instance cache will be kept around for a follow up
// while also handling intermediate incoming completion requests.
async function runReuseTestConversation(llms: LLMServer) {
	// middle part of the completion id is the instance uid.
	// we'll use this to verify which instance handled a completion.
	const messagesA: ChatMessage[] = [
		{ role: 'user', content: 'Write a haiku about bears.' },
	]
	const responseA1 = await createChatCompletion(llms, {
		model: testModel,
		temperature: 0,
		messages: messagesA,
	})
	const instanceIdA1 = getInstanceId(responseA1.completion.id)
	// do a unrelated chat completion that should be picked up by the other instance
	const responseB1 = await createChatCompletion(llms, {
		model: testModel,
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
		model: testModel,
		temperature: 0,
		messages: messagesA,
	})
	const instanceIdA2 = getInstanceId(responseA2.completion.id)
	expect(instanceIdA1).toBe(instanceIdA2)
	expect(responseA2.result.message.content).toMatch(/bear/i)
}

// conversation that tests whether reused instances leak their context
async function runResetContextConversation(llms: LLMServer) {
	const messagesA: ChatMessage[] = [
		{
			role: 'user',
			content:
				"Please remember this fact for later: Axolotls can regenerate lost limbs. Don't forget! Just answer with 'OK'.",
		},
	]
	const responseA1 = await createChatCompletion(llms, {
		model: testModel,
		temperature: 0,
		messages: messagesA,
		stop: ['OK'],
	})
	const instanceIdA1 = getInstanceId(responseA1.completion.id)
	const responseB1 = await createChatCompletion(llms, {
		model: testModel,
		temperature: 0,
		stop: ['\n'],
		messages: [{ role: 'user', content: 'Remind me of one animal fact? One Sentence.' }],
	})
	const instanceIdB1 = getInstanceId(responseB1.completion.id)
	// assert the unrelated request B1 is handled by the idle instance
	expect(instanceIdA1).not.toBe(instanceIdB1)
	// assert request B1 did not have the correct context
	expect(responseB1.result.message.content).not.toMatch(/axolotl/i)

	messagesA.push(responseA1.result.message, {
		role: 'user',
		content: 'Remind me of one animal fact? One Sentence.',
	})
	const responseA2 = await createChatCompletion(llms, {
		model: testModel,
		temperature: 0,
		stop: ['\n'],
		messages: messagesA,
	})
	const instanceIdA2 = getInstanceId(responseA2.completion.id)
	// assert follow up turn is handled by the same instance
	expect(instanceIdA1).toBe(instanceIdA2)
	// assert request A2 does not have the correct context
	expect(responseA2.result.message.content).toMatch(/axolotl/i)
}

suite('Caching behavior (node-llama-cpp)', () => {
	const llms = new LLMServer({
		inferenceConcurrency: 1,
		models: {
			[testModel]: {
				url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
				engine: 'node-llama-cpp',
				maxInstances: 2,
				templateFormat: 'phi',
			}
		},
	})
	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})
	test('Context reuse / hit', async () => {
		await runReuseTestConversation(llms)
	})
	test('Context reset / leakage', async () => {
		await runResetContextConversation(llms)
	})
})

suite('Caching behavior (gpt4all)', () => {
	const llms = new LLMServer({
		inferenceConcurrency: 1,
		models: {
			[testModel]: {
				url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
				engine: 'gpt4all',
				maxInstances: 2,
			}
		},
	})

	beforeAll(async () => {
		await llms.start()
	})

	afterAll(async () => {
		await llms.stop()
	})

	test('Context reuse / hit', async () => {
		await runReuseTestConversation(llms)
	})

	test('Context reset / leakage', async () => {
		await runResetContextConversation(llms)
	})
})
