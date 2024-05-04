import { expect } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { ChatMessage } from '#lllms/types/index.js'
import { createChatCompletion, parseInstanceId } from '../../util.js'

// conversation that tests whether the instance cache will be kept around for a follow up
// while also handling intermediate incoming completion requests.
export async function runContextReuseTest(
	llms: LLMServer,
	model: string = 'test',
) {
	// middle part of the completion id is the instance uid.
	// we'll use this to verify which instance handled a completion.
	const messagesA: ChatMessage[] = [
		{ role: 'user', content: 'Write a haiku about bears.' },
	]
	const responseA1 = await createChatCompletion(llms, {
		maxTokens: 100,
		messages: messagesA,
	})
	const instanceIdA1 = parseInstanceId(responseA1.handle.id)
	// do a unrelated chat completion that should be picked up by the other instance
	const responseB1 = await createChatCompletion(llms, {
		maxTokens: 100,
		messages: [
			{
				role: 'user',
				content: 'Write a haiku about pancakes.',
			},
		],
	})
	const instanceIdB1 = parseInstanceId(responseB1.handle.id)
	expect(instanceIdA1).not.toBe(instanceIdB1)
	// send a follow up turn to see if context is still there
	messagesA.push(responseA1.result.message, {
		role: 'user',
		content: 'Give it a 6 word title.',
	})
	const responseA2 = await createChatCompletion(llms, {
		maxTokens: 100,
		messages: messagesA,
	})
	const instanceIdA2 = parseInstanceId(responseA2.handle.id)
	expect(instanceIdA1).toBe(instanceIdA2)
	expect(responseA2.result.message.content).toMatch(/bear|giant/i)
}
