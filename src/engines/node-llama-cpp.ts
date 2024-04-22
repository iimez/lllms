import {
	getLlama,
	LlamaChatSession,
	LlamaModel,
	LlamaContext,
	ChatHistoryItem,
	ChatMLChatWrapper,
	LlamaCompletion,
} from 'node-llama-cpp'
import { LLMConfig } from '../pool'
import { ChatCompletionArgs, EngineChatCompletionResult, ChatMessage } from '../types/index.js'

interface LlamaCppInstance {
	model: LlamaModel
	context: LlamaContext
	session: LlamaChatSession | null
}

export async function loadInstance(config: LLMConfig) {
	console.debug('Creating llama instance', config)
	
	// https://github.com/withcatai/node-llama-cpp/pull/105

	const llama = await getLlama()
	const model = await llama.loadModel({
		modelPath: config.file, // full model absolute path
		// useMlock: false,
		// loadSignal: null, // cancel
	})

	const context = await model.createContext({
		// sequences: 1,
		// contextSize: "auto",
		// threads: 6, // 0 = max
		// createSignal: null, // cancel
		// seed: 0,
	})

	return {
		model,
		context,
		session: null
	}
}

export async function disposeInstance(instance: LlamaCppInstance) {
	instance.model.dispose()
}

// function getChatMessages(messages: ChatHistoryItem[]): ChatMessage[] {
// 	return messages.map((message) => {
// 		if (message.type === 'user' || message.type === 'system') {
// 			return {
// 				content: message.text,
// 				role: message.type,
// 			}
// 		}
// 		return {
// 			content: message.response.join(''),
// 			role: 'assistant',
// 		}
// 	})
// }

export async function processChatCompletion(
	instance: LlamaCppInstance,
	args: ChatCompletionArgs,
): Promise<EngineChatCompletionResult> {
	if (!instance.session) {
		instance.session = new LlamaChatSession({
			contextSequence: instance.context.getSequence(),
			chatWrapper: new ChatMLChatWrapper(),
			// systemPrompt: '',
		})
	}
	
	const input = args.messages.map((m) => m.content).join('\n')
	const result = await instance.session.promptWithMeta(input, {
		maxTokens: args.maxTokens,
		temperature: args.temperature,
		topP: args.topP,
	})
	
	// console.debug('response:', res)
	
	return {
		finishReason: result.stopReason,
		message: {
			content: result.responseText,
			role: 'assistant',
		},
		promptTokens: 0,
		completionTokens: 0,
		totalTokens: 0,
	}
}


export async function processCompletion(
	instance: LlamaCppInstance,
	args: any,
) {
	// TODO
	// const completion = new LlamaCompletion({
	// 	contextSequence: instance.context.getSequence()
	// })
	
	// const res = await completion.generateCompletionWithMeta(request.prompt, {
	// 	maxTokens: request.maxTokens,
	// })
	
	
}