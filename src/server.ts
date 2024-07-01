import os from 'node:os'
import path from 'node:path'
import { builtInEngineList } from '#lllms/engines/index.js'
import { ModelPool } from '#lllms/pool.js'
import { ModelInstance } from '#lllms/instance.js'
import { ModelStore, StoredModel } from '#lllms/store.js'
import {
	ModelOptions,
	ModelConfig,
	IncomingRequest,
	CompletionProcessingOptions,
	ChatCompletionRequest,
	EmbeddingRequest,
	ProcessingOptions,
	TextCompletionRequest,
	ModelEngine,
	ImageToTextRequest,
} from '#lllms/types/index.js'
import { Logger, LogLevel, createSublogger, LogLevels } from '#lllms/lib/logger.js'
import { resolveModelLocation } from '#lllms/lib/resolveModelLocation.js'
import { validateModelOptions } from '#lllms/lib/validation.js'

export interface ModelServerOptions {
	engines?: Record<string, ModelEngine>
	models: Record<string, ModelOptions>
	concurrency?: number
	modelsPath?: string
	log?: Logger | LogLevel
}

export function startModelServer(options: ModelServerOptions) {
	const server = new ModelServer(options)
	server.start()
	return server
}

export class ModelServer {
	pool: ModelPool
	store: ModelStore
	engines: Record<string, ModelEngine> = {}
	log: Logger

	constructor(options: ModelServerOptions) {
		this.log = createSublogger(options.log)
		const modelsPath =
			options?.modelsPath || path.resolve(os.homedir(), '.cache/lllms')

		const modelsWithDefaults: Record<string, ModelConfig> = {}
		const usedEngines: Array<{ model: string; engine: string }> = []
		for (const modelId in options.models) {
			const modelOptions = options.models[modelId]
			validateModelOptions(modelId, modelOptions)
	
			modelsWithDefaults[modelId] = {
				id: modelId,
				minInstances: 0,
				maxInstances: 1,
				engineOptions: {},
				location: resolveModelLocation(modelsPath, modelOptions),
				...modelOptions,
			}
			usedEngines.push({
				model: modelId,
				engine: modelOptions.engine,
			})
		}

		const customEngines = Object.keys(options.engines ?? {})
		for (const ref of usedEngines) {
			const isBuiltIn = builtInEngineList.includes(ref.engine)
			const isCustom = customEngines.includes(ref.engine)
			if (!isBuiltIn && !isCustom) {
				throw new Error(
					`Engine "${ref.engine}" used my model "${ref.model}" does not exist`,
				)
			}
			if (isCustom) {
				this.engines[ref.engine] = options.engines![ref.engine]
			}
		}

		this.store = new ModelStore({
			log: this.log,
			// prepareConcurrency: 2,
			models: modelsWithDefaults,
			modelsPath,
		})
		this.pool = new ModelPool(
			{
				log: this.log,
				concurrency: options.concurrency ?? 1,
				models: modelsWithDefaults,
			},
			this.prepareInstance.bind(this),
		)
	}

	modelExists(modelId: string) {
		return !!this.pool.config.models[modelId]
	}

	async start() {
		const engineStartPromises = []
		// call startEngine on custom engines
		for (const [key, methods] of Object.entries(this.engines)) {
			if (methods.start) {
				engineStartPromises.push(methods.start(this))
			}
		}
		// import built-in engines
		for (const key of builtInEngineList) {
			// skip unused engines
			const modelUsingEngine = Object.keys(this.store.models).find(
				(modelId) => this.store.models[modelId].engine === key,
			)
			if (!modelUsingEngine) {
				continue
			}
			engineStartPromises.push(
				new Promise(async (resolve, reject) => {
					try {
						const engine = await import(`#lllms/engines/${key}/engine.js`)
						this.engines[key] = engine
						resolve({
							key,
							engine,
						})
					} catch (err) {
						reject(err)
					}
				}),
			)
		}
		await Promise.all(engineStartPromises)
		await Promise.all([
			this.store.init(this.engines),
			this.pool.init(this.engines),
		])
	}
	
	async stop() {
		this.log(LogLevels.info, 'Stopping model server')
		// TODO this actually waits until all completions are done.
		// pool should be able to keep track of all running task and be able to cancel them.
		this.pool.queue.clear()
		this.store.dispose()
		await this.pool.queue.onIdle()
		await this.pool.dispose()
		this.log(LogLevels.info, 'Model server stopped')
	}

	async requestInstance(request: IncomingRequest, signal?: AbortSignal) {
		return this.pool.requestInstance(request, signal)
	}

	// gets called by the pool right before a new instance is created
	private async prepareInstance(instance: ModelInstance, signal?: AbortSignal) {
		const model = instance.config
		const modelStoreStatus = this.store.models[model.id].status
		if (modelStoreStatus === 'unloaded') {
			await this.store.prepareModel(model.id, signal)
		}
		if (modelStoreStatus === 'preparing') {
			const modelReady = new Promise<void>((resolve, reject) => {
				const onCompleted = async (storeModel: StoredModel) => {
					if (storeModel.id === model.id) {
						this.store.prepareQueue.off('completed', onCompleted)
						if (storeModel.status === 'ready') {
							resolve()
						} else {
							reject()
						}
					}
				}
				this.store.prepareQueue.on('completed', onCompleted)
			})
			await modelReady
		}
	}

	async processChatCompletionTask(
		args: ChatCompletionRequest,
		options?: CompletionProcessingOptions,
	) {
		const lock = await this.requestInstance(args)
		const task = lock.instance.processChatCompletionTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}

	async processTextCompletionTask(
		args: TextCompletionRequest,
		options?: CompletionProcessingOptions,
	) {
		const lock = await this.requestInstance(args)
		const task = lock.instance.processTextCompletionTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}

	async processEmbeddingTask(
		args: EmbeddingRequest,
		options?: ProcessingOptions,
	) {
		const lock = await this.requestInstance(args)
		const task = await lock.instance.processEmbeddingTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}
	
	async processImageToTextTask(
		args: ImageToTextRequest,
		options?: ProcessingOptions,
	) {
		const lock = await this.requestInstance(args)
		const task = await lock.instance.processImageToTextTask(args, options)
		const result = await task.result
		await lock.release()
		return result
	}



	getStatus() {
		const poolStatus = this.pool.getStatus()
		const storeStatus = this.store.getStatus()
		return {
			pool: poolStatus,
			store: storeStatus,
		}
	}
}
