import fs from 'node:fs'
import path from 'node:path'
import {
	Llama,
	LlamaGrammar,
	ChatHistoryItem,
	ChatModelResponse,
	LlamaTextJSON,
} from 'node-llama-cpp'
import { getGrammarsFolder } from 'node-llama-cpp/dist/utils/getGrammarsFolder'
import { CompletionFinishReason, ChatMessage } from '#lllms/types/index.js'
import { flattenMessageTextContent } from '#lllms/lib/flattenMessageTextContent.js'
import { LlamaChatResult } from './types.js'

export function mapFinishReason(
	nodeLlamaCppFinishReason: LlamaChatResult['stopReason'],
): CompletionFinishReason {
	switch (nodeLlamaCppFinishReason) {
		case 'functionCalls':
			return 'toolCalls'
		case 'stopGenerationTrigger':
			return 'stopTrigger'
		case 'customStopTrigger':
			return 'stopTrigger'
		default:
			return nodeLlamaCppFinishReason
	}
}

export function createSeed(min: number, max: number) {
	min = Math.ceil(min)
	max = Math.floor(max)
	return Math.floor(Math.random() * (max - min)) + min
}

export function addFunctionCallToChatHistory({
	chatHistory,
	functionName,
	functionDescription,
	callParams,
	callResult,
	rawCall,
}: {
	chatHistory: ChatHistoryItem[]
	functionName: string
	functionDescription?: string
	callParams: any
	callResult: any
	rawCall?: LlamaTextJSON
}) {
	const newChatHistory = chatHistory.slice()
	if (
		newChatHistory.length === 0 ||
		newChatHistory[newChatHistory.length - 1].type !== 'model'
	)
		newChatHistory.push({
			type: 'model',
			response: [],
		})

	const lastModelResponseItem = newChatHistory[
		newChatHistory.length - 1
	] as ChatModelResponse
	const newLastModelResponseItem = { ...lastModelResponseItem }
	newChatHistory[newChatHistory.length - 1] = newLastModelResponseItem

	const modelResponse = newLastModelResponseItem.response.slice()
	newLastModelResponseItem.response = modelResponse

	modelResponse.push({
		type: 'functionCall',
		name: functionName,
		description: functionDescription,
		params: callParams,
		result: callResult,
		rawCall,
	})

	return newChatHistory
}

export function createChatMessageArray(
	messages: ChatMessage[],
): ChatHistoryItem[] {
	const items: ChatHistoryItem[] = []
	let systemPrompt: string | undefined
	for (const message of messages) {
		if (message.role === 'user') {
			items.push({
				type: 'user',
				text: flattenMessageTextContent(message.content),
			})
		} else if (message.role === 'assistant') {
			items.push({
				type: 'model',
				response: [message.content],
			})
		} else if (message.role === 'system') {
			if (systemPrompt) {
				systemPrompt += '\n\n' + flattenMessageTextContent(message.content)
			} else {
				systemPrompt = flattenMessageTextContent(message.content)
			}
		}
	}

	if (systemPrompt) {
		items.unshift({
			type: 'system',
			text: systemPrompt,
		})
	}

	return items
}

export async function readGBNFFiles(directoryPath: string) {
	const gbnfFiles = fs
		.readdirSync(directoryPath)
		.filter((f) => f.endsWith('.gbnf'))
	const fileContents = await Promise.all(
		gbnfFiles.map((file) =>
			fs.promises.readFile(path.join(directoryPath, file), 'utf-8'),
		),
	)
	return gbnfFiles.reduce((acc, file, i) => {
		acc[file.replace('.gbnf', '')] = fileContents[i]
		return acc
	}, {} as Record<string, string>)
}


