import { expect } from 'vitest'
import { LLMServer } from '#lllms/server.js'
import { ChatMessage } from '#lllms/types/index.js'
import { createChatCompletion } from '../../util.js'

const functions = {
	getLocationWeather: {
		description: 'Get the weather in a location',
		parameters: {
			type: 'object',
			properties: {
				location: {
					type: 'string',
					description: 'The city and state, e.g. San Francisco, CA',
				},
				unit: {
					type: 'string',
					enum: ['celsius', 'fahrenheit'],
				},
			},
			required: ['location'],
		},
	},
}

export async function runFunctionCallTest(llms: LLMServer) {
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: "What's the weather like today?",
		},
	]
	const turn1 = await createChatCompletion(llms, {
		functions,
		messages,
	})
	expect(turn1.result.message.functionCalls).toBeDefined()
	expect(turn1.result.message.functionCalls!.length).toBe(1)
	console.debug({
		turn1: turn1.result.message.functionCalls,
	})

	const functionCall = turn1.result.message.functionCalls![0]
	messages.push({
		callId: functionCall.id,
		role: 'function',
		name: functionCall.name,
		// content: 'The weather is cloudy with a high chance of raining fish.',
		content: 'Today is sunny but cloudy.',
	})
	const turn2 = await createChatCompletion(llms, {
		functions,
		messages,
	})
	
	console.debug({
		turn2Response: turn2.result.message,
	})
}
