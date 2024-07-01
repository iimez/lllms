
import { EngineChatCompletionArgs, ModelEngine } from '#lllms/types/index.js'
import { CustomEngine } from '#lllms/engines/index.js'

export class ChatWithVisionEngine extends CustomEngine implements ModelEngine {
	imageToTextModel: string
	chatModel: string
	
	constructor({ imageToTextModel, chatModel }: { imageToTextModel: string, chatModel: string }) {
		super()
		this.imageToTextModel = imageToTextModel
		this.chatModel = chatModel
	}

	async processChatCompletionTask (
		args: EngineChatCompletionArgs,
	) {

		const imageTextPromises: any[] = []
		const imageToTextModel = await this.pool.requestInstance({
			model: this.imageToTextModel,
		})
		
		const messagesWithImageDescriptions = [... args.request.messages]

		for (let m = 0; m < messagesWithImageDescriptions.length; m++) {
			const message = messagesWithImageDescriptions[m]
			if (!Array.isArray(message.content)) {
				continue
			}
			for (let p = 0; p < message.content.length; p++) {
				const contentPart = message.content[p]
				if (contentPart.type !== 'image') {
					continue
				}
				imageTextPromises.push(new Promise(async (resolve, reject) => {
					const task = imageToTextModel.instance.processImageToTextTask({
						model: 'florence2',
						url: contentPart.url,
					})
					const result = await task.result
					resolve({
						text: result.text,
						messageIndex: m,
						contentPartIndex: p,
					})
				}))
			}
		}

		const imageTextResults = await Promise.all(imageTextPromises)
		
		console.debug('Image text results', imageTextResults)
		
		for (const imageTextResult of imageTextResults) {
			const { text, messageIndex, contentPartIndex } = imageTextResult
			const message = messagesWithImageDescriptions[messageIndex]
			// if ('type' in message.content[contentPartIndex]) {
			// message.content[contentPartIndex].type = 'text'
			// @ts-ignore
			message.content[contentPartIndex] = {
				type: 'text',
				text: `User uploaded image: ${text}`,
			}
		}
		const chatRequest = { ...args.request, messages: messagesWithImageDescriptions, model: this.chatModel }
		const chatModel = await this.pool.requestInstance(chatRequest)
		const task = chatModel.instance.processChatCompletionTask(chatRequest, {
			onChunk: (chunk) => {
				if (args.onChunk) args.onChunk(chunk)
			},
		})
		const result = await task.result
		chatModel.release()
		return result
	}
}