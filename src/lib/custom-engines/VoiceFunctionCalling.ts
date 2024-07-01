
import { EngineChatCompletionArgs, ModelEngine } from '#lllms/types/index.js'
import { CustomEngine } from '#lllms/engines/index.js'

// TODO

export class SpeechFunctionCalling extends CustomEngine implements ModelEngine {
	speechToTextModel: string
	chatModel: string
	
	constructor({ speechToTextModel, chatModel }: { speechToTextModel: string, chatModel: string }) {
		super()
		this.speechToTextModel = speechToTextModel
		this.chatModel = chatModel
	}
}