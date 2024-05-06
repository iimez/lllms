import { suite, it, expect, beforeAll, afterAll } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { ChatMessage, ChatCompletionRequest } from '#lllms/types/index.js'
import { createChatCompletion, parseInstanceId } from '../../util.js'

// conversation that tests behavior when context window is exceeded
export async function runContextShiftTest(
	llms: LLMServer,
	model: string = 'test',
) {
	
	// Turn 1
	const messages: ChatMessage[] = [
		{
			role: 'system',
			content: 'Be a helpful biologist.',
		},
		{
			role: 'user',
			content:
				"Please remember this fact for later: Platypuses have venomous spurs on their hind legs. Don't forget! Just answer with 'OK'.",
		},
	]
	const response1 = await createChatCompletion(llms, {
		model,
		temperature: 0,
		messages,
		stop: ['OK'],
	})
	console.debug({ response1: response1.result.message.content })
	const instanceId1 = parseInstanceId(response1.handle.id)
	
	// Turn 2
	messages.push(response1.result.message, {
		role: 'user',
		content: 'Now, I\'d like you to create a concept for a new animal. Please provide a realistic outline of a made up animal, including its social structures, appearance, habitat, origins and diet.',
	})
	const response2 = await createChatCompletion(llms, {
		temperature: 1,
		messages,
		maxTokens: 4096,
	})
	console.debug({ response2: response2.result.message.content })
	const instanceId2 = parseInstanceId(response2.handle.id)
	expect(instanceId1).toBe(instanceId2)
	
	messages.push(response2.result.message)
	
	const elaborateOn = async (field: string) => {
		messages.push({
			role: 'user',
			content: `Elaborate on its ${field}.`,
		})
		const response = await createChatCompletion(llms, {
			temperature: 1,
			messages,
			maxTokens: 4096,
		})
		console.debug({ response: response.result.message.content })
		const instanceId = parseInstanceId(response.handle.id)
		expect(instanceId1).toBe(instanceId)
		messages.push(response.result.message)
	}
	
	await elaborateOn('social structures')
	await elaborateOn('origins')
	await elaborateOn('habitat')
	await elaborateOn('appearance')
	await elaborateOn('diet')

	messages.push({
		role: 'user',
		content: 'What was the animal fact I asked you to remember earlier?',
	})
	const response3 = await createChatCompletion(llms, {
		messages,
		maxTokens: 4096,
	})
	console.debug({ response3: response3.result.message.content })
}
