import os from 'node:os'
import path from 'node:path'
import { promises as fs } from 'node:fs'
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

export interface LLMPoolOptions {
	modelsDir?: string
	concurrency?: number
	models: Record<string, LLMOptions>
}

export interface LLMPoolConfig {
	modelsDir: string
	concurrency: number
	models: Record<string, LLMConfig>
}

interface TaskResult {
	instance: LLMInstance
	args: any
}

function resolveModelFile(opts: LLMOptions, modelsDir: string) {
	if (opts.file) {
		if (path.isAbsolute(opts.file)) {
			return opts.file
		} else {
			return path.join(modelsDir, opts.file)
		}
	}

	if (opts.url) {
		const url = new URL(opts.url)
		const modelName = path.basename(url.pathname)
		const modelPath = path.join(modelsDir, modelName)
		return modelPath
	}

	throw new Error('Model file or url is required')
}

class AbortError extends Error {
	constructor(message = 'Operation aborted') {
		super(message)
		this.name = 'AbortError'
	}
}

const modelNamePattern = /^[a-zA-Z0-9_:\-]+$/
function validateModelName(modelName: string) {
	if (!modelNamePattern.test(modelName)) {
		throw new Error(`Model name must match pattern: ${modelNamePattern} (got ${modelName})`)
	}
}

type PrepareCallback = (
	instance: LLMInstance,
	signal?: AbortSignal,
) => Promise<void>

export class LLMPool extends EventEmitter3 {
	queue: PQueue
	instances: Record<string, LLMInstance>
	config: LLMPoolConfig
	pendingRequests: number = 0
	prepareInstance?: PrepareCallback

	constructor(opts: LLMPoolOptions, prepareInstance?: PrepareCallback) {
		super()
		const modelsDir =
			opts.modelsDir || path.resolve(os.homedir(), '.cache/lllms')

		const config: LLMPoolConfig = {
			modelsDir,
			concurrency: 1,
			...opts,
			models: {},
		}

		for (const modelName in opts.models) {
			validateModelName(modelName)
			const modelOpts = opts.models[modelName]
			const file = resolveModelFile(modelOpts, modelsDir)
			const modelConfig = {
				minInstances: modelOpts.minInstances || 0,
				...modelOpts,
				file,
				name: modelName,
			}
			config.models[modelName] = modelConfig
		}

		this.queue = new PQueue({ concurrency: config.concurrency })
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
		await fs.mkdir(this.config.modelsDir, { recursive: true })
		const loadingPromises = []
		const modelConfigs = this.config.models

		for (const modelName in modelConfigs) {
			const modelConfig = modelConfigs[modelName]
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
				loadPromise.then(() => this.emit('init', instance))
				loadingPromises.push(loadPromise)
			}
		}

		await Promise.all(loadingPromises)
	}

	getStatusInfo() {
		return {
			queue: this.queue.size,
			pending: this.pendingRequests,
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
							lastUse: new Date(instance.lastUse).toISOString(),
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
		this.instances[instance.id] = instance
		if (this.prepareInstance) {
			await this.prepareInstance(instance, signal)
		}
		await instance.load(signal)
		this.emit('init', instance)
	}

	modelExists(modelName: string) {
		return this.config.models[modelName] !== undefined
	}

	// modelReady(modelName: string) {
	// 	const modelInstances = Object.values(this.instances).filter(
	// 		(instance) => instance.model === modelName,
	// 	)
	// 	if (modelInstances.length > 0) {
	// 		return modelInstances.some(
	// 			(instance) => instance.status === 'idle' || instance.status === 'busy',
	// 		)
	// 	}
	// 	return false
	// }

	// acquire an instance from the pool, or create a new one if necessary
	// the instance will be locked and must be released before it can be used again
	async acquireModelInstance(modelName: string, signal?: AbortSignal) {
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
				(prev, current) => (prev.lastUse < current.lastUse ? prev : current),
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
			console.debug('spawning new instance')
			const modelConfig = this.config.models[modelName]
			const engineMethods = engines[modelConfig.engine]
			const instance = new LLMInstance(engineMethods, modelConfig)
			this.instances[instance.id] = instance
			if (this.prepareInstance) {
				await this.prepareInstance(instance, signal)
			}
			await instance.load(signal)
			instance.lock()
			return instance
		}

		// otherwise wait until an instance of our model is released
		const modelInstanceReleased = (): Promise<LLMInstance> => {
			return new Promise((resolve, reject) => {
				const onComplete = ({ instance, args }: TaskResult) => {
					// console.debug('deferred task result is', { instance, args })
					if (instance.model === args.model && instance.status === 'idle') {
						this.queue.off('completed', onComplete)
						instance.lock()
						resolve(instance)
					}
				}
				if (signal) {
					const onAbort = () => {
						this.queue.off('completed', onComplete)
						reject(new AbortError('Processing cancelled'))
					}
					signal.addEventListener('abort', onAbort)
				}
				this.queue.on('completed', onComplete)
			})
		}
		
		const modelInstanceInitialized = (): Promise<LLMInstance> => {
			return new Promise((resolve, reject) => {
				const onInit = (instance: LLMInstance) => {
					if (instance.model === modelName && instance.status === 'idle') {
						this.off('init', onInit)
						instance.lock()
						resolve(instance)
					}
				}
				if (signal) {
					const onAbort = () => {
						this.off('init', onInit)
					}
					signal.addEventListener('abort', onAbort)
				}
				this.on('init', onInit)
			})
		}

		console.debug(`waiting for an instance of ${modelName} to become available`)
		return await Promise.any([modelInstanceReleased(), modelInstanceInitialized()])
	}

	// attepts to acquire an instance that has the chat state ready, or forwards to acquireInstance
	async acquireCompletionInstance(
		args: CompletionRequest | ChatCompletionRequest,
		signal?: AbortSignal,
	) {
		// for completions first search for an instance that has the messages already ingested and the context ready
		const incomingStateHash = createContextStateHash(args, true)
		for (const key in this.instances) {
			const instance = this.instances[key]
			if (
				instance.status === 'idle' &&
				instance.model === args.model &&
				instance.contextStateHash === incomingStateHash
			) {
				console.debug('cache hit - reusing cached instance', instance.id)
				if (signal?.aborted) {
					throw new AbortError('Processing cancelled')
				}
				instance.lock()
				return instance
			}
		}

		// const relevantInstances = Object.values(this.instances).filter(
		// 	(instance) => instance.model === args.model && instance.contextStateHash && instance.status === 'idle',
		// )

		console.debug('cache miss - acquiring fresh model instance')

		return await this.acquireModelInstance(args.model, signal)
	}

	// requests an instance from the pool to handle a chat completion request
	async requestCompletionInstance(
		args: CompletionRequest | ChatCompletionRequest,
		signal?: AbortSignal,
	) {
		if (!this.config.models[args.model]) {
			throw new Error(`Model not found: ${args.model}`)
		}

		// if ('messages' in args) {
		// 	const lastMessage = args.messages[args.messages.length - 1]
		// 	console.debug('Requesting chat completion instance', lastMessage)
		// }

		// if we can, prepare a new instance to be ready for the next incoming request
		if (this.canSpawnInstance(args.model)) {
			this.spawnInstance(args.model)
		}

		this.pendingRequests++
		const instance = await this.acquireCompletionInstance(args, signal)
		this.pendingRequests--

		// once instance is acquired & locked, we can pass it on to the caller
		// the queue task promise will be forwarded as "relase" function
		let resolveTask: (value: TaskResult) => void

		this.queue.add((): Promise<TaskResult> => {
			return new Promise((resolve, reject) => {
				resolveTask = resolve
			})
		}).then((result) => {
			if (result?.instance) {
				const instance = result.instance
				console.debug(`Releasing ${instance.id}`)
				instance.unlock()
				this.emit('completed', result)
			}
			// console.debug('queue promise resolved', result.instance.id)

			// console.debug('new status', result.instance.status)
		})

		return {
			instance,
			release: () => {

				// instance.unlock()
				// console.debug('calling resolveTask')
				resolveTask({ instance, args })
			},
		}
	}
}
