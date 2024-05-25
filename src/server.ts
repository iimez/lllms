import os from 'node:os'
import path from 'node:path'
import { LLMPool } from '#lllms/pool.js'
import { LLMInstance } from '#lllms/instance.js'
import { LLMOptions, LLMConfig, IncomingLLMRequest } from '#lllms/types/index.js'
import { Logger, LogLevels, createLogger, LogLevel } from '#lllms/lib/logger.js'
import { validateModelId, resolveModelFile } from '#lllms/lib/models.js'
import { loadGBNFGrammars } from '#lllms/lib/grammar.js'
import { LLMStore } from '#lllms/store.js'

export interface LLMServerOptions {
	models: Record<string, LLMOptions>
	concurrency?: number
	maxDownloads?: number
	modelsPath?: string
	log?: Logger | LogLevel
}

export function startLLMs(opts: LLMServerOptions) {
	const server = new LLMServer(opts)
	server.start()
	return server
}

export class LLMServer {
	pool: LLMPool
	store: LLMStore
	logger: Logger

	constructor(opts: LLMServerOptions) {
		if (opts.log) {
			this.logger = typeof opts.log === 'string' ? createLogger(opts.log) : opts.log
		} else {
			this.logger = createLogger(LogLevels.warn)
		}
		const modelsPath =
			opts?.modelsPath || path.resolve(os.homedir(), '.cache/lllms')
		
		const dirname = path.dirname(new URL(import.meta.url).pathname)
		const defaultGrammars = loadGBNFGrammars(path.join(dirname, './grammars'))
		
		const modelsWithDefaults: Record<string, LLMConfig> = {}
		for (const modelId in opts.models) {
			validateModelId(modelId)
			const modelOptions = opts.models[modelId]
			if (!modelOptions.file && !modelOptions.url) {
				throw new Error(`Model ${modelId} must have either file or url`)
			}
			modelsWithDefaults[modelId] = {
				id: modelId,
				minInstances: 0,
				maxInstances: 1,
				engineOptions: {},
				...modelOptions,
				file: resolveModelFile(modelsPath, {
					file: modelOptions.file,
					url: modelOptions.url,
				}),
			}
			if (modelOptions.task === 'inference') {
				modelsWithDefaults[modelId].grammars = {
					...defaultGrammars,
					...modelOptions.grammars,
				}
			}
		}
		
		this.store = new LLMStore({
			maxDownloads: opts.maxDownloads ?? 1,
			log: this.logger,
			modelsPath,
			models: modelsWithDefaults,
		})
		this.pool = new LLMPool(
			{
				concurrency: opts.concurrency ?? 1,
				log: this.logger,
				models: modelsWithDefaults,
			},
			this.prepareInstance.bind(this),
		)
	}
	
	getModelConfig(modelId: string) {
		return this.pool.config.models[modelId]
	}

	async start() {
		await Promise.all([this.store.init(), this.pool.init()])
	}

	async requestModel(request: IncomingLLMRequest, signal?: AbortSignal) {
		return this.pool.requestInstance(request, signal)
	}

	// gets called by the pool right before a new instance is created
	private async prepareInstance(instance: LLMInstance, signal?: AbortSignal) {
		const config = instance.config
		await this.store.prepareModel(config.id, signal)
	}

	async stop() {
		// TODO this actually waits until all completions are done.
		this.pool.queue.clear()
		await this.pool.queue.onIdle()
		await this.pool.dispose()
	}

	getStatus() {
		const poolStatus = this.pool.getStatus()
		const modelStoreStatus = this.store.getStatus()
		return {
			pool: poolStatus,
			models: modelStoreStatus,
		}
	}
}

