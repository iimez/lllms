import PQueue from 'p-queue'
import EventEmitter3 from 'eventemitter3'
import { engines } from '#lllms/engines/index.js'
import { LLMInstance } from '#lllms/instance.js'
import {
	CompletionRequest,
	ChatCompletionRequest,
	LLMConfig,
	LLMRequest as IncomingLLMRequest,
} from '#lllms/types/index.js'
import { Logger, LogLevels, createLogger } from '#lllms/lib/logger.js'

interface CompletionTask {
	instance: LLMInstance
	request: CompletionRequest | ChatCompletionRequest
}

interface LLMRequestMeta {
	sequence: number
}
type LLMRequest = LLMRequestMeta & IncomingLLMRequest

type PrepareInstanceCallback = (
	instance: LLMInstance,
	signal?: AbortSignal,
) => Promise<void>

interface LLMPoolConfig {
	concurrency: number
	models: Record<string, LLMConfig>
}

export interface LLMPoolOptions {
	concurrency?: number
	releaseTimeout?: number
	models: Record<string, LLMConfig>
	logger?: Logger
}

type LLMPoolEvent = 'ready' | 'spawn' | 'release'

export class LLMPool extends EventEmitter3<LLMPoolEvent> {
	queue: PQueue
	config: LLMPoolConfig
	instances: Record<string, LLMInstance>
	private evictionInterval?: NodeJS.Timeout
	private logger: Logger
	private requestSequence: number = 1
	private waitingRequests: number = 0 // TODO should keep requests around so connections can be canceled on dispose?
	private gpuLock: boolean = false
	private prepareInstance?: PrepareInstanceCallback

	constructor(opts: LLMPoolOptions, prepareInstance?: PrepareInstanceCallback) {
		super()
		this.logger = opts.logger ?? createLogger(LogLevels.warn)
		const models: Record<string, LLMConfig> = {}
		for (const modelName in opts.models) {
			const modelConfig = opts.models[modelName]
			models[modelName] = {
				...modelConfig,
				name: modelConfig.name ?? modelName,
			}
		}
		const config: LLMPoolConfig = {
			concurrency: 1,
			...opts,
			models,
		}
		this.queue = new PQueue({
			concurrency: config.concurrency,
		})
		this.config = config
		this.instances = {}
		this.prepareInstance = prepareInstance
	}

	// start up pool, creating instances and loading models
	async init() {
		const loadingPromises = []
		const modelConfigs = this.config.models
		for (const modelName in modelConfigs) {
			// make sure name is set.
			const modelConfig = modelConfigs[modelName]
			if (!modelConfig.name) {
				modelConfig.name = modelName
			}
			// create the initial, minimum number of instances for each model
			const instanceCount = modelConfig.minInstances ?? 0
			for (let i = 0; i < instanceCount; i++) {
				if (this.canSpawnInstance(modelName)) {
					const loadPromise = this.spawnInstance(modelName)
					loadingPromises.push(loadPromise)
				} else {
					this.logger(LogLevels.warn, 'Max instances reached', {
						model: modelName,
					})
				}
			}
		}
		// resolve when all initial instances are loaded
		await Promise.all(loadingPromises)
		this.emit('ready')
		this.evictionInterval = setInterval(() => {
			this.evictOutdated()
		}, 1000 * 60) // every minute
	}

	async dispose() {
		this.logger(LogLevels.info, 'Disposing LLMPool')
		clearInterval(this.evictionInterval)
		const disposePromises = Object.values(this.instances).map((instance) =>
			instance.dispose(),
		)
		await Promise.all(disposePromises)
	}

	evictOutdated() {
		const now = new Date().getTime()

		for (const key in this.instances) {
			const instance = this.instances[key]
			const instanceAge = (now - instance.lastUsed) / 1000
			if (instanceAge > instance.ttl && instance.status === 'idle') {
				this.logger(LogLevels.info, 'Auto disposing instance', {
					instance: instance.id,
				})
				instance.dispose().then(() => {
					delete this.instances[key]
				})
			}
		}
	}

	getStatus() {
		const processingInstances = Object.values(this.instances).filter(
			(instance) => instance.status === 'busy',
		)
		return {
			processing: processingInstances.length,
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
							context: instance.getContextState(),
							lastUsed: new Date(instance.lastUsed).toISOString(),
						},
					]
				}),
			),
		}
	}

	canSpawnInstance(modelName: string) {
		const modelConfig = this.config.models[modelName]
		// if the model is configured with gpu=true, interpret that as "it MUST run on gpu"
		// and prevent spawning more instances if the gpu is already locked.
		const requiresGpu = modelConfig.engineOptions?.gpu === true
		if (requiresGpu && this.gpuLock) {
			return false
		}
		// see if we're within maxInstances limit
		const maxInstances = modelConfig.maxInstances ?? 1
		const currentInstances = Object.values(this.instances).filter(
			(instance) => instance.model === modelName,
		)
		return currentInstances.length < maxInstances
	}

	private async disposeInstance(instance: LLMInstance) {
		this.logger(LogLevels.debug, 'Disposing instance', {
			instance: instance.id,
		})
		await instance.dispose()
		delete this.instances[instance.id]
		if (instance.gpu) {
			this.gpuLock = false
		}
	}

	private async spawnInstance(
		modelName: string,
		opts: { signal?: AbortSignal; emit?: boolean } = {},
	) {
		const modelConfig = this.config.models[modelName]

		const autoGpu =
			modelConfig.engineOptions?.gpu === undefined ||
			modelConfig.engineOptions?.gpu === 'auto'
		let useGpu = autoGpu ? !this.gpuLock : false

		if (modelConfig.engineOptions?.gpu === true) {
			useGpu = true
		}

		const engineMethods = engines[modelConfig.engine]
		const instance = new LLMInstance(engineMethods, {
			...modelConfig,
			gpu: useGpu,
			logger: this.logger,
		})
		this.logger(LogLevels.info, 'Spawning instance for', {
			model: modelName,
			instance: instance.id,
			gpu: useGpu,
		})
		this.instances[instance.id] = instance

		if (useGpu) {
			this.gpuLock = true
		}
		if (this.prepareInstance) {
			await this.prepareInstance(instance, opts?.signal)
		}
		await instance.load(opts?.signal)
		if (opts.emit !== false) {
			this.emit('spawn', instance)
		}
		return instance
	}

	modelExists(modelName: string) {
		return this.config.models[modelName] !== undefined
	}

	private acquireGpuInstance(
		request: LLMRequest,
		signal?: AbortSignal,
	): Promise<LLMInstance> {
		return new Promise(async (resolve, reject) => {
			const gpuInstance = Object.values(this.instances).find(
				(instance) => instance.gpu === true,
			)!
			if (gpuInstance.status === 'idle') {
				if (gpuInstance.model === request.model) {
					gpuInstance.lock()
					resolve(gpuInstance)
					return
				} else {
					await this.disposeInstance(gpuInstance)
					const newInstance = await this.spawnInstance(request.model, {
						emit: false,
					})
					resolve(newInstance)
					return
				}
			}

			const listener = async (instance: LLMInstance) => {
				if (
					instance.matchesRequirements(request) &&
					instance.status === 'idle' &&
					instance.gpu === true
				) {
					this.off('release', listener)
					try {
						// instance.lock()
						await this.disposeInstance(instance)
						const newInstance = await this.spawnInstance(request.model, {
							emit: false,
						})
						resolve(newInstance)
					} catch (err: any) {
						this.logger(LogLevels.error, 'Error acquiring gpu instance', {
							error: err.message,
						})
						reject(err)
					}
				}
			}
			// this.on('spawn', listener)
			this.on('release', listener)
			if (signal) {
				signal.addEventListener('abort', () => {
					this.off('release', listener)
					// this.off('spawn', listener)
					reject(signal.reason)
				})
			}
		})
	}

	// wait to acquire an idle instance for the given request
	private acquireIdleInstance(
		request: LLMRequest,
		signal?: AbortSignal,
	): Promise<LLMInstance> {
		return new Promise((resolve, reject) => {
			const listener = (instance: LLMInstance) => {
				if (
					instance.matchesRequirements(request) &&
					instance.status === 'idle'
				) {
					this.off('release', listener)
					this.off('spawn', listener)
					try {
						instance.lock()
						resolve(instance)
					} catch (err: any) {
						this.logger(LogLevels.error, 'Error acquiring idle instance', {
							error: err.message,
						})
						reject(err)
					}
				}
			}
			this.on('spawn', listener)
			this.on('release', listener)
			if (signal) {
				signal.addEventListener('abort', () => {
					this.off('release', listener)
					this.off('spawn', listener)
					reject(signal.reason)
				})
			}
		})
	}

	private async acquireInstance(request: LLMRequest, signal?: AbortSignal) {
		if ('messages' in request) {
			// for chat completions first search for an instance that has the messages already ingested and the context ready
			for (const key in this.instances) {
				const instance = this.instances[key]
				if (
					instance.matchesRequirements(request) &&
					instance.status === 'idle' &&
					instance.matchesContextState(request)
				) {
					this.logger(LogLevels.debug, 'Cache hit - reusing cached instance', {
						instance: instance.id,
						sequence: request.sequence,
					})
					instance.lock()
					return instance
				}
			}
			this.logger(
				LogLevels.debug,
				'Cache miss - acquiring fresh model instance',
				{ sequence: request.sequence },
			)
		}

		// prefer an instance of the model that has no context state.
		for (const key in this.instances) {
			const instance = this.instances[key]
			if (
				instance.matchesRequirements(request) &&
				instance.status === 'idle' &&
				!instance.hasContextState()
			) {
				this.logger(LogLevels.debug, 'Reusing instance with no context state', {
					instance: instance.id,
					sequence: request.sequence,
				})
				instance.lock()
				return instance
			}
		}

		// if all instances have cached state, prefer the one that was used the longest time ago
		const availableInstances = Object.values(this.instances).filter(
			(instance) =>
				instance.matchesRequirements(request) && instance.status === 'idle',
		)
		if (availableInstances.length > 0) {
			const lruInstance = availableInstances.reduce((prev, current) =>
				prev.lastUsed < current.lastUsed ? prev : current,
			)
			this.logger(LogLevels.debug, 'Reusing least recently used instance', {
				instance: lruInstance.id,
				sequence: request.sequence,
			})
			lruInstance.lock()
			lruInstance.resetContext() // make sure we reset its cache.
			return lruInstance
		}

		// still havent found any, see if we're allowed to spawn a new instance
		if (this.canSpawnInstance(request.model)) {
			const instance = await this.spawnInstance(request.model, {
				emit: false,
			})
			this.logger(LogLevels.debug, 'Spawned instance acquired', {
				instance: instance.id,
				sequence: request.sequence,
			})
			instance.lock()
			return instance
		}

		const requiresGpu =
			this.config.models[request.model].engineOptions?.gpu === true
		if (requiresGpu && this.gpuLock) {
			const gpuInstance = Object.values(this.instances).find(
				(instance) => instance.gpu === true,
			)!

			if (gpuInstance.model !== request.model) {
				const instance = await this.acquireGpuInstance(request, signal)
				this.logger(LogLevels.debug, 'GPU instance acquired', {
					instance: instance.id,
					sequence: request.sequence,
				})
				if (signal?.aborted) {
					instance.unlock()
					throw signal.reason
				} else {
					return instance
				}
			}
		}

		// otherwise wait until an instance of our model is released or spawned
		this.logger(LogLevels.debug, 'Awaiting idle instance for', {
			model: request.model,
			sequence: request.sequence,
		})
		const instance = await this.acquireIdleInstance(request, signal)
		this.logger(LogLevels.debug, 'Idle instance acquired', {
			instance: instance.id,
			sequence: request.sequence,
		})

		if (signal?.aborted) {
			instance.unlock()
			throw signal.reason
		} else {
			return instance
		}
	}

	// requests an instance from the pool to handle a chat completion request
	async requestLLM(incomingRequest: IncomingLLMRequest, signal?: AbortSignal) {
		const request = {
			...incomingRequest,
			sequence: this.requestSequence++,
		}
		if (!this.config.models[request.model]) {
			this.logger(LogLevels.error, `Model not found: ${request.model}`)
			throw new Error(`Model not found: ${request.model}`)
		}

		this.logger(LogLevels.info, 'Incoming request for', {
			model: request.model,
			sequence: request.sequence,
		})

		// if we can, prepare a new instance to be ready for the next incoming request
		// TODO this slows down the response time for all requests until maxInstances is reached.
		// should do it after the request is completed. only spawn a new instance during completion
		// processing if theres actually another request.
		if (this.canSpawnInstance(request.model)) {
			this.spawnInstance(request.model)
		}

		this.waitingRequests++
		const instance = await this.acquireInstance(request, signal)

		// once instance is acquired & locked, we can pass it on to the caller
		// the queue task promise will be forwarded as releaseInstance
		let resolveQueueTask: (value: CompletionTask) => void = () => {}

		this.queue
			.add((): Promise<CompletionTask> => {
				this.waitingRequests--
				return new Promise((resolve, reject) => {
					resolveQueueTask = resolve
				})
			})
			.then((task) => {
				if (task?.instance) {
					const instance = task.instance
					this.logger(LogLevels.info, 'Task completed, releasing instance', {
						instance: instance.id,
						sequence: request.sequence,
					})
					instance.unlock()
					this.emit('release', instance)
				}
			})

		// TODO what if user never calls release? automatically resolve or reject after a timeout?
		const releaseInstance = () => {
			resolveQueueTask({ instance, request })
		}
		return {
			instance,
			release: releaseInstance,
		}
	}
}
