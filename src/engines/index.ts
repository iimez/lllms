import * as gpt4all from './gpt4all.js'
import * as nodeLlamaCpp from './node-llama-cpp.js'

export const engines = {
	gpt4all: gpt4all,
	'node-llama-cpp': nodeLlamaCpp,
} as const

export type EngineType = keyof typeof engines
