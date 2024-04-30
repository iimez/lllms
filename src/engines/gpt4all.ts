import path from 'node:path'
import {
	loadModel,
	createCompletion,
	InferenceModel,
	LoadModelOptions,
	CompletionInput,
} from 'gpt4all'
import {
	LLMConfig,
	ChatCompletionRequest,
	CompletionRequest,
	EngineCompletionContext,
	EngineChatCompletionResult,
	EngineCompletionResult,
	CompletionFinishReason,
	ChatTemplateFormat,
	EngineContext,
} from '../types/index.js'
import { LogLevels } from '../util/log.js'

export async function loadInstance(config: LLMConfig, ctx: EngineContext) {
	ctx.logger(LogLevels.info, `Load GPT4All model from ${config.file}`, {
		instance: ctx.instance,
	})

	const loadOpts: LoadModelOptions = {
		modelPath: path.dirname(config.file),
		// file: config.file,
		// allowDownload: false,
		device: config.gpu ? 'gpu' : 'cpu',
		// nCtx: 2048,
		// ngl: 100,
		// verbose: true,
	}
	// console.debug('creating gpt4all model instance', {
	// 	modelName,
	// 	loadOpts,
	// })
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
	ctx: EngineCompletionContext,
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
			if (ctx.onChunk) {
				ctx.onChunk({
					text: token,
					tokens: [tokenId],
				})
			}
			return !ctx?.signal?.aborted
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

function addSystemPromptTemplate(
	systemPrompt: string,
	templateFormat: ChatTemplateFormat,
) {
	if (templateFormat === 'chatml') {
		return `<|im_start|>system\n${systemPrompt}<|im_end|>\n`
	}
	if (templateFormat === 'llama3') {
		return `<|start_header_id|>system<|end_header_id|>\n\n${systemPrompt}<|eot_id|>`
	}
	if (templateFormat === 'alpaca') {
		return `### System:\n${systemPrompt}\n\n`
	}
	if (templateFormat === 'phi') {
		return `<|system|>\n${systemPrompt}<|end|>\n`
	}
	if (templateFormat === 'llama2') {
		// return `[INST]${systemPrompt}[/INST]\n`
		return `<s>[INST] <<SYS>>\n${systemPrompt}\n<</SYS>>[/INST]</s>\n\n`
	}
	return systemPrompt
}

export async function processChatCompletion(
	instance: InferenceModel,
	request: ChatCompletionRequest,
	ctx: EngineCompletionContext,
): Promise<EngineChatCompletionResult> {
	let session = instance.activeChatSession
	if (!session || ctx.resetContext) {
		let systemPrompt = request.systemPrompt

		// allow setting system prompt via initial message.
		if (!systemPrompt && request.messages[0].role === 'system') {
			systemPrompt = request.messages[0].content
			if (request.templateFormat) {
				systemPrompt = addSystemPromptTemplate(
					systemPrompt,
					request.templateFormat,
				)
			}
		}
		// console.debug('using system prompt', systemPrompt)
		session = await instance.createChatSession({
			systemPrompt,
		})
	}

	// if we have reset context, we need to reingest the chat history,
	// or otherwise just append the last user message.
	const nonSystemMessages = request.messages.filter((m) => m.role !== 'system')
	let input: CompletionInput

	if (ctx.resetContext) {
		// reingests all, then prompts automatically for last user message
		input = nonSystemMessages
	} else {
		// append the last (user) message
		const lastMessage = nonSystemMessages[nonSystemMessages.length - 1]
		if (lastMessage.role !== 'user') {
			throw new Error('Last message must be from user.')
		}
		input = lastMessage.content
	}

	let finishReason: CompletionFinishReason = 'eogToken'

	const result = await createCompletion(session, input, {
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
			if (request.stop && request.stop.includes(token)) {
				finishReason = 'stopGenerationTrigger'
				return false
			}
			if (ctx.onChunk) {
				ctx.onChunk({
					tokens: [tokenId],
					text: token,
				})
			}

			return !ctx?.signal?.aborted
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
