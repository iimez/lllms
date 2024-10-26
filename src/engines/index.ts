import type { ModelPool } from '#package/pool.js'
import type { ModelStore } from '#package/store.js'
import { ModelEngine, EngineStartContext, ModelOptions, BuiltInModelOptions } from '#package/types/index.js'

export const BuiltInEngines = {
	gpt4all: 'gpt4all',
	nodeLlamaCpp: 'node-llama-cpp',
	transformersJs: 'transformers-js',
	stableDiffusionCpp: 'stable-diffusion-cpp',
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
