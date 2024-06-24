import { suite, it, expect, beforeAll, afterAll } from 'vitest'
import fs from 'node:fs'
import { LLMServer } from '#lllms/server.js'
import { ChatMessage } from '#lllms/types/index.js'
import { createChatCompletion } from '../../util.js'

// conversation that tests behavior when context window is exceeded while model is ingesting text
export async function runContextShiftIngestionTest(
	llms: LLMServer,
	model: string = 'test',
) {
	
	const text = fs.readFileSync('tests/fixtures/dunwich.txt', 'utf-8')
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: text + text + '\n\nWhats that?',
		},
	]
	const response1 = await createChatCompletion(llms, {
		model,
		messages,
		maxTokens: 512,
	})
	
	// TODO get instance from llms and check what happened
	console.debug({ response1: response1.result.message.content })
}
