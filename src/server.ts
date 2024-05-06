import os from 'node:os'
import path from 'node:path'
import { LLMPool } from '#lllms/pool.js'
import { LLMInstance } from '#lllms/instance.js'
import { LLMOptions, LLMRequest } from '#lllms/types/index.js'
import { Logger, LogLevels, createLogger, LogLevel } from '#lllms/lib/logger.js'
import { LLMStore } from '#lllms/store.js'

export interface LLMServerOptions {
	models: Record<string, LLMOptions>
	inferenceConcurrency?: number
	downloadConcurrency?: number
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
		this.store = new LLMStore(modelsPath, opts.models, {
			downloadConcurrency: opts.downloadConcurrency ?? 1,
		})
		this.pool = new LLMPool(
			{
				inferenceConcurrency: opts.inferenceConcurrency ?? 1,
				log: this.logger,
				models: this.store.models,
			},
			this.prepareInstance.bind(this),
		)
	}

	async start() {
		await Promise.all([this.store.init(), this.pool.init()])
	}

	async requestLLM(request: LLMRequest, signal?: AbortSignal) {
		return this.pool.requestLLM(request, signal)
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

