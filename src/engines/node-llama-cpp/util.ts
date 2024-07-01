import fs from 'node:fs'
import path from 'node:path'
import {
	Llama,
	LlamaGrammar,
	ChatHistoryItem,
	ChatModelResponse,
	LlamaTextJSON,
} from 'node-llama-cpp'
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

export function prepareGrammars(
	llama: Llama,
	grammarConfig: Record<string, string>,
) {
	const grammars: Record<string, LlamaGrammar> = {}
	for (const key in grammarConfig) {
		const grammar = new LlamaGrammar(llama, {
			grammar: grammarConfig[key],
			// printGrammar: true,
		})
		grammars[key] = grammar
	}
	return grammars
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

export function readGrammarFiles(grammarsPath: string) {
	const gbnfFiles = fs
		.readdirSync(grammarsPath)
		.filter((f) => f.endsWith('.gbnf'))
	const grammars: Record<string, string> = {}
	for (const file of gbnfFiles) {
		const grammar = fs.readFileSync(path.join(grammarsPath, file), 'utf-8')
		grammars[file.replace('.gbnf', '')] = grammar
	}
	return grammars
}

