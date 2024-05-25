import type { LLMServer } from '#lllms/server.js'
import { createChatCompletionHandler } from './handlers/chat.js'
import { createCompletionHandler } from './handlers/completions.js'
import { createModelsHandler } from './handlers/models.js'
import { createEmbeddingsHandler } from './handlers/embeddings.js'


// See OpenAI API specs at https://github.com/openai/openai-openapi/blob/master/openapi.yaml
export function createOpenAIRequestHandlers(llmServer: LLMServer) {
	return {
		chatCompletions: createChatCompletionHandler(llmServer),
		completions: createCompletionHandler(llmServer),
		models: createModelsHandler(llmServer),
		embeddings: createEmbeddingsHandler(llmServer),
	}
}
