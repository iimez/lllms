import fs from 'node:fs'
import { expect } from 'vitest'
import { ModelServer } from '#lllms/server.js'
import { ChatMessage } from '#lllms/types/index.js'
import { createChatCompletion } from '../../util.js'

// conversation that tests behavior when context window is exceeded while model is ingesting text
export async function runContextShiftIngestionTest(
	llms: ModelServer,
	model: string = 'test',
) {
	
	const text = fs.readFileSync('tests/fixtures/lovecraft.txt', 'utf-8')
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: text + text + '\n\nWhats that? End your explanation with "OK".',
		},
	]
	const response1 = await createChatCompletion(llms, {
		model,
		messages,
		maxTokens: 512,
	})
	expect(response1.result.message.content?.substring(-6)).toMatch(/OK/i)
	// TODO get instance from llms and check what happened
	console.debug({ response1: response1.result.message.content })

}
