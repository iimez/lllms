import type { ModelPool } from '#lllms/pool.js'
import type { ModelStore } from '#lllms/store.js'
import { ModelEngine, EngineStartContext, ModelOptions, BuiltInModelOptions } from '#lllms/types/index.js'
// export type { NodeLlamaCppEngineOptions } from './node-llama-cpp/engine.js'
// export type { GPT4AllEngineOptions } from './gpt4all/engine.js'
// export type { TransformersJsEngineOptions } from './transformers-js/engine.js'

export const BuiltInEngines = {
	gpt4all: 'gpt4all',
	nodeLlamaCpp: 'node-llama-cpp',
	transformersJs: 'transformers-js',
} as const

export type BuiltInEngineName = typeof BuiltInEngines[keyof typeof BuiltInEngines];

export const builtInEngineNames: string[] = [
	...Object.values(BuiltInEngines),
] as const

export class CustomEngine implements ModelEngine {
	pool!: ModelPool
	store!: ModelStore
	async start({ pool, store }: EngineStartContext) {
		this.pool = pool
		this.store = store
	}
	async prepareModel() {}
	async createInstance() {}
	async disposeInstance() {}
}

export class NestedEngine implements ModelEngine {
	pool!: ModelPool
	store!: ModelStore
	async start() {
		// this.pool = pool
		// this.store = store
	}
	async prepareModel() {}
	async createInstance() {
		
	}
	async disposeInstance() {}
}
