import PQueue from 'p-queue'
import EventEmitter3 from 'eventemitter3'
import { engines } from '#lllms/engines/index.js'
import { LLMInstance } from '#lllms/instance.js'
import {
	LLMConfig,
	IncomingLLMRequest,
	LLMRequest,
} from '#lllms/types/index.js'
import { Logger, LogLevels, createLogger, LogLevel } from '#lllms/lib/logger.js'

export interface LLMInstanceHandle {
	instance: LLMInstance
	release: () => Promise<void>
}

interface LLMTask {
	instance: LLMInstance
	request: LLMRequest
}

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
	models: Record<string, LLMConfig>
	log?: Logger | LogLevel
}

type LLMPoolEvent = 'ready' | 'spawn' | 'release'

export class LLMPool extends EventEmitter3<LLMPoolEvent> {
	queue: PQueue
	config: LLMPoolConfig
	instances: Record<string, LLMInstance>
	private evictionInterval?: NodeJS.Timeout
	private log: Logger
	private requestSequence: number = 0
	private waitingRequests: number = 0 // TODO should keep requests around so connections can be canceled on dispose?
	private gpuLock: boolean = false // TODO could derive this from "is there any instance that has gpu=true"
	private prepareInstance?: PrepareInstanceCallback

	constructor(opts: LLMPoolOptions, prepareInstance?: PrepareInstanceCallback) {
		super()
		if (opts.log) {
			this.log =
				typeof opts.log === 'string' ? createLogger(opts.log) : opts.log
		} else {
			this.log = createLogger(LogLevels.warn)
		}
		const models: Record<string, LLMConfig> = {}
		for (const id in opts.models) {
			const modelConfig = opts.models[id]
			models[id] = {
				...modelConfig,
				id: modelConfig.id ?? id,
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
		const initPromises = []
		const modelConfigs = this.config.models

		// making sure id is set.
		for (const modelId in modelConfigs) {
			const modelConfig = modelConfigs[modelId]
			if (!modelConfig.id) {
				modelConfig.id = modelId
			}
		}

		// prioritize initializing the first model defined that has gpu=true
		// so lock cant be acquired first by another model that has gpu=auto/undefined
		const firstGpuModel = Object.entries(modelConfigs).find(
			([id, config]) => config.engineOptions?.gpu === true,
		)
		if (firstGpuModel) {
			const modelConfig = modelConfigs[firstGpuModel[0]]
			const spawnPromises = this.ensureModelInstances(modelConfig)
			initPromises.push(...spawnPromises)
		}

		// then handle other models in the order they were defined
		for (const modelId in modelConfigs) {
			if (firstGpuModel && modelId === firstGpuModel[0]) {
				continue
			}
			const modelConfig = modelConfigs[modelId]
			const spawnPromises = this.ensureModelInstances(modelConfig)
			initPromises.push(...spawnPromises)
		}

		// resolve when all initial instances are loaded
		await Promise.allSettled(initPromises)
		this.emit('ready')
		this.evictionInterval = setInterval(() => {
			this.evictOutdated()
		}, 1000 * 60) // every minute
	}

	// see if the minInstances for a models are spawned. if not, spawn them.
	ensureModelInstances(model: LLMConfig) {
		const spawnPromises = []
		const instanceCount = model.minInstances ?? 0
		for (let i = 0; i < instanceCount; i++) {
			if (this.canSpawnInstance(model.id)) {
				const spawnPromise = this.spawnInstance(model.id)
				spawnPromises.push(spawnPromise)
			} else {
				this.log(LogLevels.warn, 'Failed to spawn min instances for', {
					model: model.id,
				})
				break
			}
		}
		return spawnPromises
	}

	async dispose() {
		this.log(LogLevels.info, 'Disposing LLMPool')
		clearInterval(this.evictionInterval)
		const disposePromises = Object.values(this.instances).map((instance) =>
			instance.dispose(),
		)
		await Promise.all(disposePromises)
	}

	// disposes instances that have been idle for longer than their ttl
	private evictOutdated() {
		const now = new Date().getTime()
		for (const key in this.instances) {
			const instance = this.instances[key]
			const instanceAge = (now - instance.lastUsed) / 1000
			const modelInstanceCount = Object.values(this.instances).filter(
				(i) => i.model === instance.model,
			).length
			const minInstanceCount =
				this.config.models[instance.model].minInstances ?? 0
			if (
				modelInstanceCount > minInstanceCount &&
				instanceAge > instance.ttl &&
				instance.status === 'idle'
			) {
				this.log(LogLevels.info, 'Auto disposing instance', {
					instance: instance.id,
				})
				this.disposeInstance(instance).then(() => {
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
							device: instance.gpu ? 'gpu' : 'cpu',
							context: instance.getContextState(),
							lastUsed: new Date(instance.lastUsed).toISOString(),
						},
					]
				}),
			),
		}
	}

	// checks if another instance can be spawned for given model
	canSpawnInstance(modelId: string) {
		const modelConfig = this.config.models[modelId]
		// if the model is configured with gpu=true, interpret that as "it MUST run on gpu"
		// and prevent spawning more instances if the gpu is already locked.
		const requiresGpu = modelConfig.engineOptions?.gpu === true
		if (requiresGpu && this.gpuLock) {
			// TODO check if we are allowed to shut down the locking instance
			// this.log(
			// 	LogLevels.debug,
			// 	'Denied spawning instance because of GPU lock',
			// 	{
			// 		model: modelId,
			// 	},
			// )
			return false
		}
		// see if we're within maxInstances limit
		const maxInstances = modelConfig.maxInstances ?? 1
		const currentInstances = Object.values(this.instances).filter(
			(instance) => instance.model === modelId,
		)
		if (currentInstances.length >= maxInstances) {
			// this.log(
			// 	LogLevels.debug,
			// 	'Denied spawning instance because of maxInstances',
			// 	{
			// 		model: modelId,
			// 	},
			// )
			return false
		}
		return true
	}

	private async disposeInstance(instance: LLMInstance) {
		this.log(LogLevels.debug, 'Disposing instance', {
			instance: instance.id,
		})
		await instance.dispose()
		if (instance.gpu) {
			this.gpuLock = false
		}
		delete this.instances[instance.id]
	}

	// spawns a new instance for the given model, without checking whether it's allowed
	private async spawnInstance(
		modelId: string,
		options: { signal?: AbortSignal; emit?: boolean } = {},
	) {
		const model = this.config.models[modelId]

		// if the model is configured with gpu=auto (or unset), we can use the gpu if its not locked
		const autoGpu =
			model.engineOptions?.gpu === undefined ||
			model.engineOptions?.gpu === 'auto'
		let useGpu = autoGpu ? !this.gpuLock : false

		if (model.engineOptions?.gpu === true) {
			useGpu = true
		}

		const engineMethods = engines[model.engine]
		const instance = new LLMInstance(engineMethods, {
			...model,
			gpu: useGpu,
			logger: this.log,
		})
		this.instances[instance.id] = instance

		if (useGpu) {
			this.gpuLock = true
		}
		if (this.prepareInstance) {
			this.log(LogLevels.debug, 'Preparing instance', {
				instance: instance.id,
			})
			try {
				await this.prepareInstance(instance, options?.signal)
			} catch (error) {
				this.log(LogLevels.error, 'Error preparing instance', {
					model: modelId,
					instance: instance.id,
					error,
				})
				instance.status = 'error'
				return instance
			}
		}
		await instance.load(options?.signal)
		if (options.emit !== false) {
			this.emit('spawn', instance)
		}
		return instance
	}

	// wait to acquire a gpu instance for the given request
	private acquireGpuInstance(
		request: LLMRequest,
		signal?: AbortSignal,
	): Promise<LLMInstance> {
		return new Promise(async (resolve, reject) => {
			// if we have an idle gpu instance and the model matches we can lock and return immediately
			const gpuInstance = Object.values(this.instances).find(
				(instance) => instance.gpu === true,
			)!
			if (gpuInstance.status === 'idle') {
				if (gpuInstance.model === request.model) {
					gpuInstance.lock(request)
					resolve(gpuInstance)
					return
				} else {
					await this.disposeInstance(gpuInstance)

					const newInstance = await this.spawnInstance(request.model, {
						emit: false,
					})
					newInstance.lock(request)
					resolve(newInstance)
					return
				}
			}

			// otherwise attach the listener and wait until gpu slot becomes available
			const listener = async (instance: LLMInstance) => {
				if (
					instance.matchesRequirements(request) &&
					instance.status === 'idle' &&
					instance.gpu === true
				) {
					this.off('release', listener)
					await this.disposeInstance(instance)
					const newInstance = await this.spawnInstance(request.model, {
						emit: false,
					})
					newInstance.lock(request)
					resolve(newInstance)
				}
			}
			this.on('release', listener)
			if (signal) {
				signal.addEventListener('abort', () => {
					this.off('release', listener)
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
						instance.lock(request)
						resolve(instance)
					} catch (error: any) {
						this.log(LogLevels.error, 'Error acquiring idle instance', {
							error,
						})
						reject(error)
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

	// acquire an instance for the given request
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
					this.log(LogLevels.debug, 'Cache hit - reusing cached instance', {
						instance: instance.id,
						sequence: request.sequence,
					})
					instance.lock(request)
					return instance
				}
			}
			this.log(
				LogLevels.debug,
				'Cache miss - continue acquiring model instance',
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
				this.log(LogLevels.debug, 'Reusing instance with no context state', {
					instance: instance.id,
					sequence: request.sequence,
				})
				instance.lock(request)
				return instance
			}
		}

		// still havent found any, see if we're allowed to spawn a new instance
		if (this.canSpawnInstance(request.model)) {
			const instance = await this.spawnInstance(request.model, {
				emit: false,
			})
			this.log(LogLevels.debug, 'Spawned instance acquired', {
				instance: instance.id,
				sequence: request.sequence,
			})
			instance.lock(request)
			return instance
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
			this.log(LogLevels.debug, 'Reusing least recently used instance', {
				instance: lruInstance.id,
				sequence: request.sequence,
			})
			lruInstance.lock(request)
			lruInstance.reset() // make sure we reset its cache.
			return lruInstance
		}

		const requiresGpu =
			this.config.models[request.model].engineOptions?.gpu === true
		if (requiresGpu && this.gpuLock) {
			const gpuInstance = Object.values(this.instances).find(
				(instance) => instance.gpu === true,
			)!

			if (gpuInstance.model !== request.model) {
				this.log(LogLevels.debug, 'Awaiting GPU instance for', {
					model: request.model,
					sequence: request.sequence,
				})
				const instance = await this.acquireGpuInstance(request, signal)
				this.log(LogLevels.debug, 'GPU instance acquired', {
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

		// before starting to wait, make sure sure we're not stuck with an error'd instance (and wait forever)
		// currently instances only enter error state if prepareInstance throws an error
		const errorInstance = Object.values(this.instances).find(
			(instance) =>
				instance.model === request.model && instance.status === 'error',
		)
		if (errorInstance) {
			throw new Error('Instance is in error state')
		}

		// wait until an instance of our model is released or spawned
		this.log(LogLevels.debug, 'Awaiting idle instance for', {
			model: request.model,
			sequence: request.sequence,
		})
		const instance = await this.acquireIdleInstance(request, signal)
		this.log(LogLevels.debug, 'Idle instance acquired', {
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

	private getRequestSequence() {
		if (this.requestSequence > 999999) {
			this.requestSequence = 0
		}
		return ++this.requestSequence
	}

	// requests an language model instance from the pool
	async requestInstance(
		incomingRequest: IncomingLLMRequest,
		signal?: AbortSignal,
	): Promise<LLMInstanceHandle> {
		const requestSequence = this.getRequestSequence()
		const request = {
			...incomingRequest,
			sequence: requestSequence,
		}
		if (!this.config.models[request.model]) {
			this.log(LogLevels.error, `Model not found: ${request.model}`)
			throw new Error(`Model not found: ${request.model}`)
		}

		this.log(LogLevels.info, 'Incoming request for', {
			model: request.model,
			sequence: request.sequence,
		})

		this.waitingRequests++
		const instance = await this.acquireInstance(request, signal)

		// once instance is acquired & locked, we can pass it on to the caller
		// the queue task promise will be forwarded as releaseInstance
		let resolveQueueTask: (value: LLMTask) => void = () => {}

		this.queue
			.add((): Promise<LLMTask> => {
				this.waitingRequests--
				return new Promise((resolve, reject) => {
					resolveQueueTask = resolve
				})
			})
			.then((task) => {
				// if theres more requests waiting, prioritize handling them first
				if (!this.waitingRequests && this.canSpawnInstance(request.model)) {
					this.spawnInstance(request.model)
				}
				if (task?.instance) {
					this.emit('release', instance)
				}
			})

		// TODO what if user never calls release? automatically resolve or reject after a timeout?
		const releaseInstance = () => {
			return new Promise<void>((resolve, reject) => {
				process.nextTick(() => {
					resolveQueueTask({ instance, request })
					this.log(LogLevels.info, 'Task completed, releasing', {
						instance: instance.id,
						sequence: request.sequence,
					})
					instance.unlock()
					resolve()
				})
			})
		}

		return {
			instance,
			release: releaseInstance,
		}
	}
}
