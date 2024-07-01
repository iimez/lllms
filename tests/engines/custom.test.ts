import { suite, it, test, beforeAll, afterAll, expect } from 'vitest'
// @ts-ignore
import { Florence2ForConditionalGeneration } from '@xenova/transformers'
import { ModelServer } from '#lllms/server.js'
import {
	ChatCompletionRequest,
	ChatMessage,
	ModelEngine,
	ModelOptions,
} from '#lllms/types/index.js'
import { ChatWithVisionEngine } from '#lllms/lib/custom-engines/ChatWithVision.js'
import { createChatCompletion } from '../util'

const engines: Record<string, ModelEngine> = {
	'chat-with-vision': new ChatWithVisionEngine({
		chatModel: 'llama3-8b',
		imageToTextModel: 'florence2',
	}),
}

const models: Record<string, ModelOptions> = {
	test: {
		engine: 'chat-with-vision',
		task: 'text-completion',
	},
	'llama3-8b': {
		url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
		md5: 'c87ad09e1e4c8f9c35a5fcef52b6f1c9',
		engine: 'gpt4all',
		task: 'text-completion',
	},
	florence2: {
		url: 'https://huggingface.co/onnx-community/Florence-2-large-ft',
		engine: 'transformers-js',
		task: 'image-to-text',
		// minInstances: 1,
		engineOptions: {
			modelClass: Florence2ForConditionalGeneration,
			dtype: {
				embed_tokens: 'fp16',
				vision_encoder: 'fp32',
				encoder_model: 'fp16',
				decoder_model_merged: 'q4',
			},
		},
	},
}

suite('basic', () => {
	const llms = new ModelServer({
		// log: 'debug',
		engines,
		models,
	})

	beforeAll(async () => {
		await llms.start()
	})
	afterAll(async () => {
		await llms.stop()
	})

	it('can see', async () => {
		const messages: ChatMessage[] = [
			{
				role: 'user',
				content: [
					{
						type: 'image',
						url: 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true',
					},
					{
						type: 'text',
						text: 'WHAT DO YOUR ELF EYES SEE?',
					},
				],
			},
		]
		const response = await createChatCompletion(llms, {
			temperature: 0,
			messages,
		})
		console.debug({ response: response.result.message.content })
	})
	
	it('can hear', async () => {
			// TODO
	})
})
