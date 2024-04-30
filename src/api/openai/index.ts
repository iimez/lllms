import { LLMPool } from '../../pool.js'
import { createChatCompletionHandler } from './routes/chat.js'
import { createCompletionHandler } from './routes/completions.js'
import { createListModelsHandler } from './routes/models.js'

// See OpenAI API specs at https://github.com/openai/openai-openapi/blob/master/openapi.yaml
export function createOpenAIRequestHandlers(pool: LLMPool) {
	return {
		chatCompletions: createChatCompletionHandler(pool),
		completions: createCompletionHandler(pool),
		listModels: createListModelsHandler(pool),
	}
}
