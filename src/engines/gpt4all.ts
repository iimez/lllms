import crypto from 'node:crypto'
import path from 'node:path'
import { loadModel, createCompletion, InferenceModel, LoadModelOptions } from 'gpt4all'
import { LLMConfig,  } from '../pool.js'
import { ChatCompletionArgs, EngineChatCompletionResult, ChatMessage } from '../types/index.js'

export async function loadInstance(config: LLMConfig) {
	const modelName = config.name
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
	return instance
}

export async function disposeInstance(instance: InferenceModel) {
	return instance.dispose()
}

// export function onChatCompletionRequest(
// 	instance: InferenceModel,
// 	args: ChatCompletionArgs,
// ): ChatCompletionRequest {
// 	if (!instance.activeChatSession) {
// 		return {
// 			...request,
// 			cached: false,
// 		}
// 	}

// 	const incomingMessages = [...request.messages]
// 	const lastMessage = incomingMessages.pop()
// 	const incomingStateHash = createConversationHash(incomingMessages)
// 	const currentStateHash = createConversationHash(instance.activeChatSession?.messages)
// 	const requestMatchesState = incomingStateHash === currentStateHash

// 	if (requestMatchesState && lastMessage) {
// 		return {
// 			...request,
// 			cached: true,
// 			messages: [lastMessage],
// 		}
// 	}

// 	return {
// 		...request,
// 		cached: false,
// 	}
// }

export async function processChatCompletion(
	instance: InferenceModel,
	args: ChatCompletionArgs,
): Promise<EngineChatCompletionResult> {

	let session = instance.activeChatSession
	if (!session) {
		
		let systemPrompt = args.systemPrompt
		if (!systemPrompt && args.messages[0].role === 'system') {
			systemPrompt = args.messages[0].content

		}
		systemPrompt = `<|im_start|>system\n${systemPrompt}\n<|im_end|>`
		// systemPrompt = `<|start_header_id|>system<|end_header_id|>\n\n${systemPrompt}<|eot_id|>`
		console.debug('using system prompt', systemPrompt)
		session = await instance.createChatSession({
			systemPrompt: args.systemPrompt,
		})
	}
	
	const nonSystemMessages = args.messages.filter((m) => m.role !== 'system')

	const result = await createCompletion(session, nonSystemMessages, {
		temperature: args.temperature,
		nPredict: args.maxTokens,
		// seed: args.seed,
		// frequencyPenalty: args.frequencyPenalty,
		// presencePenalty: args.presencePenalty,
		// topK: 40,
		topP: args.topP,
		// minP: args.minP,
		// promptTemplate:
		// nBatch: 8,
		// repeatPenalty: 1.18,
		// repeatLastN: 10,
	})
	
	return {
		finishReason: "stopGenerationTrigger", // not available
		message: {
			role: 'assistant',
			content: result.choices[0].message.content,
		},
		promptTokens: result.usage.prompt_tokens,
		completionTokens: result.usage.completion_tokens,
		totalTokens: result.usage.total_tokens,
	}
}
