import { suite, it, expect, beforeAll, afterAll } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { ChatMessage, ChatCompletionRequest } from '#lllms/types/index.js'
import { createChatCompletion, parseInstanceId } from '../../util.js'

// conversation that tests whether reused instances leak their context
export async function runContextLeakTest(
	llms: LLMServer,
	model: string = 'test',
) {
	const messagesA: ChatMessage[] = [
		{
			role: 'user',
			content:
				"Please remember this fact for later: Axolotls can regenerate lost limbs. Don't forget! Just answer with 'OK'.",
		},
	]
	const responseA1 = await createChatCompletion(llms, {
		model,
		temperature: 0,
		messages: messagesA,
		maxTokens: 64,
		stop: ['OK'],
	})
	const instanceIdA1 = parseInstanceId(responseA1.handle.id)
	console.debug({ responseA1: responseA1.result.message.content })
	const responseB1 = await createChatCompletion(llms, {
		stop: ['\n'],
		maxTokens: 100,
		messages: [
			{ role: 'user', content: 'Remind me of one animal fact? One Sentence.' },
		],
	})
	// console.debug({ responseB1: responseB1.result.message.content })
	const instanceIdB1 = parseInstanceId(responseB1.handle.id)
	// assert the unrelated request B1 is handled by the idle instance
	expect(instanceIdA1).not.toBe(instanceIdB1)
	// assert request B1 did not have the correct context
	expect(responseB1.result.message.content).not.toMatch(/axolotl/i)

	messagesA.push(responseA1.result.message, {
		role: 'user',
		content: 'Remind me of one animal fact? One Sentence.',
	})
	const responseA2 = await createChatCompletion(llms, {
		stop: ['\n'],
		messages: messagesA,
		maxTokens: 100,
	})
	const instanceIdA2 = parseInstanceId(responseA2.handle.id)
	// assert follow up turn is handled by the same instance
	expect(instanceIdA1).toBe(instanceIdA2)
	// assert request A2 does not have the correct context
	expect(responseA2.result.message.content).toMatch(/axolotl/i)
}
