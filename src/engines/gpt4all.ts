import path from 'node:path'
import {
	loadModel,
	createCompletion,
	InferenceModel,
	LoadModelOptions,
} from 'gpt4all'
import {
	LLMConfig,
	CompletionRequest,
	GenerationArgs,
	EngineChatCompletionResult,
	EngineCompletionResult,
	CompletionFinishReason,
	ChatTemplateFormat,
} from '../types/index.js'

export async function loadInstance(config: LLMConfig, signal?: AbortSignal) {
	const modelName = config.name
	const loadOpts: LoadModelOptions = {
		modelPath: path.dirname(config.file),
		// file: config.file,
		// allowDownload: false,
		device: config.gpu ? 'gpu' : 'cpu',
		// nCtx: 2048,
		// ngl: 100,
		// verbose: true,
	}
	console.debug('creating gpt4all model instance', {
		modelName,
		loadOpts,
	})
	// TODO no way to cancel it / use signal
	const instance = await loadModel(path.basename(config.file), loadOpts)
	return instance
}

export async function disposeInstance(instance: InferenceModel) {
	return instance.dispose()
}

export async function processCompletion(
	instance: InferenceModel,
	request: CompletionRequest,
	args: GenerationArgs,
): Promise<EngineCompletionResult> {
	if (!request.prompt) {
		throw new Error('Prompt is required for completion.')
	}

	let finishReason: CompletionFinishReason = 'eogToken'

	const result = await instance.generate(request.prompt, {
		temperature: request.temperature,
		nPredict: request.maxTokens,
		topP: request.topP,
		promptTemplate: '%1',
		// supported by gpt4all but dont exist openai
		// @ts-ignore
		// special: true, // allows passing in raw prompt (including <|start|> etc.)
		// minP: args.minP,
		// nBatch: 8,
		// repeatPenalty: 1.18,
		// repeatLastN: 10,
		// other openai options that dont exist in gpt4all
		// seed: args.seed,
		// frequencyPenalty: args.frequencyPenalty,
		// presencePenalty: args.presencePenalty,
		// topK: 40,
		onResponseToken: (tokenId, token) => {
			if (request.stop && request.stop.includes(token)) {
				finishReason = 'stopGenerationTrigger'
				return false
			}
			if (args.onChunk) {
				args.onChunk({
					token: token,
					tokenId,
				})
			}
			return !args.signal?.aborted
		},
	})

	if (result.tokensGenerated === request.maxTokens) {
		finishReason = 'maxTokens'
	}

	return {
		finishReason,
		text: result.text,
		promptTokens: result.tokensIngested,
		completionTokens: result.tokensGenerated,
		totalTokens: result.tokensIngested + result.tokensGenerated,
	}
}

function addSystemPromptTemplate(systemPrompt: string, templateFormat: ChatTemplateFormat) {
	if (templateFormat === 'chatml') {
		return `<|im_start|>system\n${systemPrompt}<|im_end|>\n`
	}
	if (templateFormat === 'llama3') {
		return `<|start_header_id|>system<|end_header_id|>\n\n${systemPrompt}<|eot_id|>`
	}
	if (templateFormat === 'alpaca') {
		return `### System:\n${systemPrompt}\n\n`
	}
	return systemPrompt

}

export async function processChatCompletion(
	instance: InferenceModel,
	request: CompletionRequest,
	args: GenerationArgs,
): Promise<EngineChatCompletionResult> {
	if (!request.messages) {
		throw new Error('Messages are required for chat completion.')
	}

	let session = instance.activeChatSession
	if (!session) {
		let systemPrompt = request.systemPrompt
		if (!systemPrompt && request.messages[0].role === 'system') {
			systemPrompt = request.messages[0].content
			if (request.templateFormat) {
				systemPrompt = addSystemPromptTemplate(systemPrompt, request.templateFormat)
			}
		}

		// console.debug('using system prompt', systemPrompt)
		session = await instance.createChatSession({
			systemPrompt,
		})
	}

	const nonSystemMessages = request.messages.filter(
		(m) => m.role !== 'system',
	)
	let finishReason: CompletionFinishReason = 'eogToken'

	const result = await createCompletion(session, nonSystemMessages, {
		temperature: request.temperature,
		nPredict: request.maxTokens,
		topP: request.topP,
		// supported by gpt4all but dont exist openai
		// @ts-ignore
		// special: true, // allows passing in raw prompt (including <|start|> etc.)
		// minP: args.minP,
		// nBatch: 8,
		// repeatPenalty: 1.18,
		// repeatLastN: 10,
		// other openai options that dont exist in gpt4all
		// seed: args.seed,
		// frequencyPenalty: args.frequencyPenalty,
		// presencePenalty: args.presencePenalty,
		// topK: 40,
		onResponseToken: (tokenId, token) => {
			if (args.onChunk) {
				args.onChunk({
					tokenId,
					token: token,
				})
			}

			return !args.signal?.aborted
		},
	})
	
	if (result.usage.completion_tokens === request.maxTokens) {
		finishReason = 'maxTokens'
	}

	return {
		finishReason,
		message: {
			role: 'assistant',
			content: result.choices[0].message.content,
		},
		promptTokens: result.usage.prompt_tokens,
		completionTokens: result.usage.completion_tokens,
		totalTokens: result.usage.total_tokens,
	}
}
