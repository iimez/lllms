import {
	ChatModelFunctions,
	LlamaChatResponse,
	LlamaChatResponseFunctionCall,
} from 'node-llama-cpp'

export interface LlamaChatResult<T extends ChatModelFunctions = any> {
	responseText: string | null
	functionCalls?: LlamaChatResponseFunctionCall<T>[]
	stopReason: LlamaChatResponse['metadata']['stopReason']
}
