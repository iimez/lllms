import PQueue from 'p-queue'
import EventEmitter3 from 'eventemitter3'
import { engines } from './engines/index.js'
import { LLMInstance } from './instance.js'
import {
	CompletionRequest,
	ChatCompletionRequest,
	LLMConfig,
	LLMOptions,
} from './types/index.js'
import { createContextStateHash } from './util/createContextStateHash.js'

class AbortError extends Error {
	constructor(message = 'Operation aborted') {
		super(message)
		this.name = 'AbortError'
	}
}


interface CompletionTask {
	instance: LLMInstance
	req: CompletionRequest | ChatCompletionRequest
}

type PrepareInstanceCallback = (
	instance: LLMInstance,
	signal?: AbortSignal,
) => Promise<void>

interface LLMPoolConfig {
	inferenceConcurrency: number
	models: Record<string, LLMConfig>
}

export interface LLMPoolModelOptions extends LLMOptions {
	file: string
	name?: string
}

export interface LLMPoolOptions {
	concurrency?: number
	models: Record<string, LLMPoolModelOptions>
}

export class LLMPool extends EventEmitter3 {
	queue: PQueue
	config: LLMPoolConfig
	waitingRequests: number = 0
	instances: Record<string, LLMInstance>
	prepareInstance?: PrepareInstanceCallback

	constructor(
		opts: LLMPoolOptions,
		prepareInstance?: PrepareInstanceCallback,
	) {
		super()

		const models: Record<string, LLMConfig> = {}
		for (const modelName in opts.models) {
			const modelConfig = opts.models[modelName]
			models[modelName] = {
				...modelConfig,
				name: modelConfig.name ?? modelName,
			}
		}
		const config: LLMPoolConfig = {
			inferenceConcurrency: 1,
			...opts,
			models,
		}
		this.queue = new PQueue({
			concurrency: config.inferenceConcurrency
		})
		this.config = config
		this.instances = {}
		this.prepareInstance = prepareInstance
	}

	async dispose() {
		const disposePromises = Object.values(this.instances).map((instance) =>
			instance.dispose(),
		)
		await Promise.all(disposePromises)
	}

	// start up pool, creating instances and loading models
	async init() {
		const loadingPromises = []
		const modelConfigs = this.config.models

		for (const modelName in modelConfigs) {
			const modelConfig = modelConfigs[modelName]
			if (!modelConfig.name) {
				modelConfig.name = modelName
			}
			const engineMethods = engines[modelConfig.engine]

			// create the initial, minimum number of instances for each model
			const instanceCount = modelConfig.minInstances ?? 0
			for (let i = 0; i < instanceCount; i++) {
				const instance = new LLMInstance(engineMethods, modelConfig)
				this.instances[instance.id] = instance
				if (this.prepareInstance) {
					await this.prepareInstance(instance)
				}
				const loadPromise = instance.load()
				loadPromise.then(() => this.emit('spawn', instance))
				loadingPromises.push(loadPromise)
			}
		}

		await Promise.all(loadingPromises)
	}

	getStatusInfo() {
		return {
			processing: this.queue.size,
			waiting: this.waitingRequests,
			instances: Object.fromEntries(
				Object.entries(this.instances).map(([key, instance]) => {
					return [
						key,
						{
							model: instance.model,
							status: instance.status,
							url: instance.config.url,
							file: instance.config.file,
							engine: instance.config.engine,
							context: instance.contextStateHash,
							lastUsed: new Date(instance.lastUsed).toISOString(),
						},
					]
				}),
			),
		}
	}

	private canSpawnInstance(modelName: string) {
		const modelConfig = this.config.models[modelName]
		const maxInstances = modelConfig.maxInstances ?? 1
		const currentInstances = Object.values(this.instances).filter(
			(instance) => instance.model === modelName,
		)
		return currentInstances.length < maxInstances
	}

	private async spawnInstance(modelName: string, signal?: AbortSignal) {
		const modelConfig = this.config.models[modelName]
		const engineMethods = engines[modelConfig.engine]
		const instance = new LLMInstance(engineMethods, modelConfig)
		console.debug(`${instance.id} spawning`)
		this.instances[instance.id] = instance
		if (this.prepareInstance) {
			await this.prepareInstance(instance, signal)
		}
		await instance.load(signal)
		this.emit('spawn', instance)
		return instance
	}

	modelExists(modelName: string) {
		return this.config.models[modelName] !== undefined
	}
	
	// wait to acquire the next idle instance of the given model.
	// if this gets called multiple times, promises will resolve in the order they were called.
	acquireIdleInstance(modelName: string, signal?: AbortSignal): Promise<LLMInstance> {
		return new Promise((resolve, reject) => {
			const listener = (instance: LLMInstance) => {
				if (instance.model === modelName && instance.status === 'idle') {
					this.off('release', listener)
					this.off('spawn', listener)
					instance.lock()
					resolve(instance)
				}
			}
			this.on('spawn', listener)
			this.on('release', listener)
			if (signal) {
				signal.addEventListener('abort', () => {
					this.off('release', listener)
					this.off('spawn', listener)
					reject(new AbortError())
				})
			}
		})
	}

	// acquire an instance from the pool, or create a new one if possible, or wait until one is available
	// the instance will be locked and must be released before it can be used again
	async acquireInstance(modelName: string, signal?: AbortSignal) {
		// prefer an instance of the model that has no context state.
		for (const key in this.instances) {
			const instance = this.instances[key]
			if (
				instance.status === 'idle' &&
				instance.model === modelName &&
				!instance.contextStateHash
			) {
				console.debug('reusing instance with no context state', instance.id)
				instance.lock()
				return instance
			}
		}

		// if all instances have cached state, prefer the one that was used the longest time ago
		const availableInstances = Object.values(this.instances).filter(
			(instance) => instance.status === 'idle' && instance.model === modelName,
		)
		if (availableInstances.length > 0) {
			const leastRecentlyUsedInstance = availableInstances.reduce(
				(prev, current) => (prev.lastUsed < current.lastUsed ? prev : current),
			)
			console.debug(
				'reusing least recently used instance',
				leastRecentlyUsedInstance.id,
			)
			leastRecentlyUsedInstance.lock()
			leastRecentlyUsedInstance.resetContext() // make sure we reset its cache.
			return leastRecentlyUsedInstance
		}

		// see if we're allowed to spawn a new instance
		if (this.canSpawnInstance(modelName)) {
			const instance = await this.spawnInstance(modelName, signal)
			instance.lock()
			return instance
		}

		// otherwise wait until an instance of our model is released or spawned
		console.debug(`awaiting instance of ${modelName} ...`)

		const instance = await this.acquireIdleInstance(modelName)
		
		console.debug(`${instance.id} acquired`)

		if (signal?.aborted) {
			instance.unlock()
			throw new AbortError()
		} else {
			return instance
		}
	}

	// attepts to acquire an instance that has the chat state ready, or forwards to acquireInstance
	async acquireCompletionInstance(
		req: CompletionRequest | ChatCompletionRequest,
		signal?: AbortSignal,
	) {
		// for completions first search for an instance that has the messages already ingested and the context ready
		const incomingStateHash = createContextStateHash(req, true)
		for (const key in this.instances) {
			const instance = this.instances[key]
			if (
				instance.status === 'idle' &&
				instance.model === req.model &&
				instance.contextStateHash === incomingStateHash
			) {
				console.debug('cache hit - reusing cached instance', instance.id)
				instance.lock()
				return instance
			}
		}

		console.debug('cache miss - acquiring fresh model instance')
		return await this.acquireInstance(req.model, signal)
	}

	// requests an instance from the pool to handle a chat completion request
	async requestCompletionInstance(
		req: CompletionRequest | ChatCompletionRequest,
		signal?: AbortSignal,
	) {
		if (!this.config.models[req.model]) {
			throw new Error(`Model not found: ${req.model}`)
		}

		// if ('messages' in args) {
		// 	const lastMessage = args.messages[args.messages.length - 1]
		// 	console.debug('Requesting chat completion instance', lastMessage)
		// }

		// if we can, prepare a new instance to be ready for the next incoming request
		// TODO this slows down the response time for all requests until maxInstances is reached.
		// should do it after the request is completed. only spawn a new instance during completion
		// processing if theres actually another request.
		if (this.canSpawnInstance(req.model)) {
			this.spawnInstance(req.model)
		}

		this.waitingRequests++
		const instance = await this.acquireCompletionInstance(req, signal)
		this.waitingRequests--

		// once instance is acquired & locked, we can pass it on to the caller
		// the queue task promise will be forwarded as releaseInstance
		let resolveQueueTask: (value: CompletionTask) => void

		this.queue
			.add((): Promise<CompletionTask> => {
				return new Promise((resolve, reject) => {
					resolveQueueTask = resolve
				})
			})
			.then((task) => {
				console.debug('task resolved')
				if (task?.instance) {
					const instance = task.instance
					console.debug(`${instance.id} releasing`)
					instance.unlock()
					console.debug(`${instance.id} now`, instance.status)
					this.emit('release', instance)
				}
			})

		// TODO what if user never calls release? automatically resolve or reject after a timeout?
		const releaseInstance = () => {
			console.debug('calling resolveQueueTask')
			resolveQueueTask({ instance, req })
		}
		return {
			instance,
			releaseInstance,
			// releaseInstance: () => {
			// 	console.debug('calling resolveQueueTask')
			// 	resolveQueueTask({ instance, req })
			// }
		}
	}
}
