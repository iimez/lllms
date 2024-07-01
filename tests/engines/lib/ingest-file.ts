import fs from 'node:fs'
import { ModelServer } from '#lllms/server.js'
import { ChatMessage } from '#lllms/types/index.js'
import { createChatCompletion } from '../../util.js'

export async function runFileIngestionTest(
	llms: ModelServer,
	file: string,
	prompt: string = 'Whats that?',
	model: string = 'test',
) {
	const text = fs.readFileSync(`tests/fixtures/${file}.txt`, 'utf-8')
	const messages: ChatMessage[] = [
		{
			role: 'user',
			content: text + '\n---\n\n' + prompt,
		},
	]
	const response = await createChatCompletion(llms, {
		model,
		messages,
		maxTokens: 256,
	})
	return response.result
}