import { LLMEngine } from '#lllms/types/index.js'
import * as gpt4all from './gpt4all.js'
import * as nodeLlamaCpp from './node-llama-cpp.js'

export type { LlamaCppOptions } from './node-llama-cpp.js'
export type { GPT4AllOptions } from './gpt4all.js'

export const engines: Record<string, LLMEngine> = {
	gpt4all: gpt4all,
	'node-llama-cpp': nodeLlamaCpp,
} as const

export type EngineType = keyof typeof engines
export type EngineInstance = any