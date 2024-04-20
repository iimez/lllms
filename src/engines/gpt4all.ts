import crypto from 'node:crypto'
import path from 'node:path'
import { loadModel, createCompletion, InferenceModel, LoadModelOptions } from 'gpt4all'
import { LLMConfig, ChatCompletionRequest, ChatMessage } from '../pool'

function createConversationHash(messages: ChatMessage[]): string {
	return crypto.createHash('sha256').update(JSON.stringify(messages)).digest('hex')
}

export async function createInstance(config: LLMConfig) {
	const modelName = config.name
	// const modelsDir = process.env.MODELS_DIR || 'models'
	// const modelPath = path.join(modelsDir, modelName)
	const loadOpts: LoadModelOptions = {
		modelPath: path.dirname(config.file),
		// file: config.file,
		// allowDownload: false,
		device: 'cpu',
		// nCtx: 2048,
		// ngl: 100,
		// verbose: true,
	}
	console.debug('creating gpt4all model instance', {
		modelName,
		loadOpts,
	})
	const instance = await loadModel(path.basename(config.file), loadOpts)
	// await instance.createChatSession()

	return instance
}

export async function disposeInstance(instance: InferenceModel) {
	return instance.dispose()
}

export function onChatCompletionRequest(
	instance: InferenceModel,
	request: ChatCompletionRequest,
): ChatCompletionRequest {
	if (!instance.activeChatSession) {
		return {
			...request,
			cached: false,
		}
	}

	const incomingMessages = [...request.messages]
	const lastMessage = incomingMessages.pop()
	const incomingStateHash = createConversationHash(incomingMessages)
	const currentStateHash = createConversationHash(instance.activeChatSession?.messages)
	const requestMatchesState = incomingStateHash === currentStateHash

	if (requestMatchesState && lastMessage) {
		return {
			...request,
			cached: true,
			messages: [lastMessage],
		}
	}

	return {
		...request,
		cached: false,
	}
}

export async function processChatCompletion(
	instance: InferenceModel,
	req: ChatCompletionRequest,
) {
	if (instance.activeChatSession) {
		return await createCompletion(instance.activeChatSession, req.messages)
	}
	const session = await instance.createChatSession()
	return await createCompletion(session, req.messages)
}
