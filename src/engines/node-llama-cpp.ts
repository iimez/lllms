import crypto from 'node:crypto'
// import { loadModel, createCompletion, InferenceModel } from 'gpt4all'
// import {LlamaModel, LlamaContext, LlamaChatSession} from "node-llama-cpp";
import {
	getLlama,
	LlamaChatSession,
	LlamaModel,
	LlamaContext,
	ChatHistoryItem,
	// ChatMLChatPromptWrapper,
	ChatMLChatWrapper
} from 'node-llama-cpp'
import { LLMConfig, ChatCompletionRequest, ChatMessage } from '../pool'

// https://github.com/withcatai/node-llama-cpp/pull/105

interface LlamaCppInstanceWrap {
	model: LlamaModel
	context: LlamaContext
	session: LlamaChatSession
}

export async function createInstance(config: LLMConfig) {
	console.debug('Creating llama instance', config)

	const llama = await getLlama()
	const model = await llama.loadModel({
		modelPath: config.file,
		// modelPath: path.join(__dirname, "models", "dolphin-2.1-mistral-7b.Q4_K_M.gguf")
	})

	const context = await model.createContext()
	const session = new LlamaChatSession({
		contextSequence: context.getSequence(),
		chatWrapper: new ChatMLChatWrapper(),
	})

	// @ts-ignore
	// const model = new LlamaModel({
	// 		// modelPath: path.join(__dirname, "models", "codellama-13b.Q3_K_M.gguf")
	// });
	// // @ts-ignore
	// const context = new LlamaContext({model});

	// const session = new LlamaChatSession({context});

	return {
		model,
		context,
		session,
	}
}

export async function disposeInstance(instance: LlamaCppInstanceWrap) {
	// instance.dispose()
	instance.model.dispose()
}

function createConversationHash(messages: ChatMessage[]): string {
	return crypto
		.createHash('sha256')
		.update(JSON.stringify(messages))
		.digest('hex')
}

function getChatMessages(messages: ChatHistoryItem[]): ChatMessage[] {
	return messages.map((message) => {
		if (message.type === 'user' || message.type === 'system') {
			return {
				content: message.text,
				role: message.type,
			}
		}
		return {
			content: message.response.join(''),
			role: 'assistant',
		}
	})
}

export function onChatCompletionRequest(
	instance: LlamaCppInstanceWrap,
	request: ChatCompletionRequest,
) {
	const incomingMessages = [...request.messages]
	const lastMessage = incomingMessages.pop()
	const incomingStateHash = createConversationHash(incomingMessages)
	const currentMessages = getChatMessages(instance.session.getChatHistory())
	const currentStateHash = createConversationHash(currentMessages)
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
	instance: LlamaCppInstanceWrap,
	request: ChatCompletionRequest,
) {
	// if (!instance.activeChatSession) {
	// 	await instance.createChatSession()
	// }

	// return await createCompletion(instance.activeChatSession, req.messages)

	const res = await instance.session.prompt(
		request.messages.map((m) => m.content).join('\n'),
	)
	
	// instance.session.chatWrapper.
	
	console.debug('Chat completion response:', res)
}
