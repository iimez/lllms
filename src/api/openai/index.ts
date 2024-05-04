import type { LLMPool } from '#lllms/pool.js'
import { createChatCompletionHandler } from './handlers/chat.js'
import { createCompletionHandler } from './handlers/completions.js'
import { createListModelsHandler } from './handlers/models.js'

// See OpenAI API specs at https://github.com/openai/openai-openapi/blob/master/openapi.yaml
export function createOpenAIRequestHandlers(pool: LLMPool) {
	return {
		chatCompletions: createChatCompletionHandler(pool),
		completions: createCompletionHandler(pool),
		listModels: createListModelsHandler(pool),
	}
}
