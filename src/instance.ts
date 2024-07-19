import crypto from 'node:crypto'
import { customAlphabet } from 'nanoid'
import {
	TextCompletionRequest,
	ChatCompletionRequest,
	ModelEngine,
	ModelConfig,
	ModelInstanceRequest,
	CompletionProcessingOptions,
	EmbeddingRequest,
	ImageToTextRequest,
	ProcessingOptions,
	SpeechToTextRequest,
	SpeechToTextProcessingOptions,
} from '#lllms/types/index.js'
import { calculateChatIdentity } from '#lllms/lib/calculateChatIdentity.js'
import {
	LogLevels,
	Logger,
	createLogger,
	withLogMeta,
} from '#lllms/lib/logger.js'
import { elapsedMillis, mergeAbortSignals } from '#lllms/lib/util.js'

const idAlphabet =
	'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
const generateId = customAlphabet(idAlphabet, 8)

type ModelInstanceStatus = 'idle' | 'busy' | 'error' | 'loading' | 'preparing'

interface ModelInstanceOptions extends ModelConfig {
	log?: Logger
	gpu: boolean
}

export class ModelInstance<TEngineState = unknown> {
	id: string
	status: ModelInstanceStatus
	modelId: string
	config: ModelConfig
	fingerprint: string
	createdAt: Date
	lastUsed: number = 0
	gpu: boolean
	ttl: number
	log: Logger

	private engine: ModelEngine
	private contextStateIdentity?: string
	private needsContextReset: boolean = false
	private engineInstance?: TEngineState | unknown
	private currentRequest?: ModelInstanceRequest | null
	private shutdownController: AbortController

	constructor(
		engine: ModelEngine,
		{ log, gpu, ...options }: ModelInstanceOptions,
	) {
		this.modelId = options.id
		this.id = this.generateInstanceId()
		this.engine = engine
		this.config = options
		this.gpu = gpu
		this.ttl = options.ttl ?? 300
		this.status = 'preparing'
		this.createdAt = new Date()
		this.log = withLogMeta(log ?? createLogger(LogLevels.warn), {
			instance: this.id,
		})
		this.shutdownController = new AbortController()

		// TODO to implement this properly we should only include what changes the "behavior" of the model
		this.fingerprint = crypto
			.createHash('sha1')
			.update(JSON.stringify(options))
			.digest('hex')
		this.log(LogLevels.info, 'Initializing new instance', {
			model: this.modelId,
			engine: this.config.engine,
			device: this.config.device,
			hasGpuLock: this.gpu,
		})
	}

	private generateInstanceId() {
		return this.modelId + ':' + generateId(8)
	}

	private generateTaskId() {
		return this.id + '-' + generateId(8)
	}

	async load(signal?: AbortSignal) {
		if (this.engineInstance) {
			throw new Error('Instance is already loaded')
		}
		this.status = 'loading'
		const loadBegin = process.hrtime.bigint()
		const abortSignal = mergeAbortSignals([
			this.shutdownController.signal,
			signal,
		])
		try {
			this.engineInstance = await this.engine.createInstance(
				{
					log: withLogMeta(this.log, {
						instance: this.id,
					}),
					config: {
						...this.config,
						device: {
							...this.config.device,
							gpu: this.gpu,
						},
					},
				},
				abortSignal,
			)
			this.status = 'idle'
			if (this.config.preload) {
				if ('messages' in this.config.preload) {
					this.contextStateIdentity = calculateChatIdentity(
						this.config.preload.messages,
					)
				}
			}
			this.log(LogLevels.debug, 'Instance loaded', {
				elapsed: elapsedMillis(loadBegin),
			})
		} catch (error: any) {
			this.status = 'error'
			this.log(LogLevels.error, 'Failed to load instance:', {
				error,
			})
			throw error
		}
	}

	dispose() {
		this.status = 'busy'
		if (!this.engineInstance) {
			return Promise.resolve()
		}
		this.shutdownController.abort()
		return this.engine.disposeInstance(this.engineInstance)
	}

	lock(request: ModelInstanceRequest) {
		if (this.status !== 'idle') {
			throw new Error(`Cannot lock: Instance ${this.id} is not idle`)
		}
		this.currentRequest = request
		this.status = 'busy'
	}

	unlock() {
		this.status = 'idle'
		this.currentRequest = null
	}

	reset() {
		this.needsContextReset = true
	}

	getContextStateIdentity() {
		return this.contextStateIdentity
	}

	hasContextState() {
		return this.contextStateIdentity !== undefined
	}

	matchesContextState(request: ModelInstanceRequest) {
		if (!this.contextStateIdentity) {
			return false
		}
		if (!('messages' in request)) {
			return false
		}
		const incomingStateIdentity = calculateChatIdentity(request.messages, true)
		return this.contextStateIdentity === incomingStateIdentity
	}

	matchesRequirements(request: ModelInstanceRequest) {
		const mustGpu = this.config.device?.gpu === true
		const modelMatches = this.modelId === request.model
		const gpuMatches = mustGpu ? this.gpu : true
		return modelMatches && gpuMatches
	}

	private createTaskController(args: {
		timeout?: number
		signal?: AbortSignal
	}) {
		const cancelController = new AbortController()
		const timeoutController = new AbortController()
		const abortSignals = [cancelController.signal, this.shutdownController.signal]
		if (args.signal) {
			abortSignals.push(args.signal)
		}
		let timeout: NodeJS.Timeout | undefined
		if (args.timeout) {
			timeout = setTimeout(() => {
				timeoutController.abort()
			}, args.timeout)
			abortSignals.push(timeoutController.signal)
		}
		return {
			cancel: () => {
				cancelController.abort()
			},
			complete: () => {
				if (timeout) {
					clearTimeout(timeout)
				}
			},
			signal: mergeAbortSignals(abortSignals),
			timeoutSignal: timeoutController.signal,
		}
	}

	processChatCompletionTask(
		request: ChatCompletionRequest,
		options?: CompletionProcessingOptions,
	) {
		if (!('processChatCompletionTask' in this.engine)) {
			throw new Error(
				`Engine "${this.config.engine}" does not implement chat completions`,
			)
		}
		if (!request.messages?.length) {
			throw new Error('Messages are required for chat completions')
		}
		const id = this.generateTaskId()
		this.lastUsed = Date.now()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		// checking if this instance has been flagged for reset
		let resetContext = false
		if (this.needsContextReset) {
			this.contextStateIdentity = undefined
			this.needsContextReset = false
			resetContext = true
		}
		const controller = this.createTaskController({
			timeout: options?.timeout,
			signal: options?.signal,
		})
		// start completion processing
		taskLogger(LogLevels.verbose, 'Creating chat completion')
		const taskBegin = process.hrtime.bigint()
		const completionPromise = this.engine.processChatCompletionTask!(
			{
				request,
				resetContext,
				config: this.config,
				log: taskLogger,
				onChunk: options?.onChunk,
			},
			this.engineInstance,
			controller.signal,
		).then((result) => {
			const elapsedTime = elapsedMillis(taskBegin)
			controller.complete()
			if (controller.timeoutSignal.aborted) {
				taskLogger(LogLevels.warn, 'Chat completion task timed out')
				result.finishReason = 'timeout'
			}
			this.contextStateIdentity = calculateChatIdentity([
				...request.messages,
				result.message,
			])
			taskLogger(LogLevels.info, 'Chat completion done', {
				elapsed: elapsedTime,
			})
			return result
		}).catch((error) => {
			taskLogger(LogLevels.error, 'Task failed - ', {
				error,
			})
			throw error
		})
		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			result: completionPromise,
			cancel: controller.cancel,
		}
	}

	processTextCompletionTask(
		request: TextCompletionRequest,
		options?: CompletionProcessingOptions,
	) {
		if (!('processTextCompletionTask' in this.engine)) {
			throw new Error(
				`Engine "${this.config.engine}" does not implement text completion`,
			)
		}
		if (!request.prompt) {
			throw new Error('Prompt is required for text completion')
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: options?.timeout,
			signal: options?.signal,
		})
		taskLogger(LogLevels.verbose, 'Creating text completion task')
		const taskBegin = process.hrtime.bigint()
		const completionPromise = this.engine.processTextCompletionTask!(
			{
				request,
				config: this.config,
				log: taskLogger,
				onChunk: options?.onChunk,
			},
			this.engineInstance,
			controller.signal,
		).then((result) => {
			// TODO allow continueing / caching prefix for text completions?
			// this.contextStateHash = calculateChatIdentity({
			// 	prompt: request.prompt,
			// })
			const elapsedTime = elapsedMillis(taskBegin)
			controller.complete()
			if (controller.timeoutSignal.aborted) {
				taskLogger(LogLevels.warn, 'Text completion task timed out')
				result.finishReason = 'timeout'
			}
			taskLogger(LogLevels.verbose, 'Text completion task done', {
				elapsed: elapsedTime,
			})
			return result
		}).catch((error) => {
			taskLogger(LogLevels.error, 'Task failed - ', {
				error,
			})
			throw error
		})
		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result: completionPromise,
		}
	}

	processEmbeddingTask(request: EmbeddingRequest, options?: ProcessingOptions) {
		if (!('processEmbeddingTask' in this.engine)) {
			throw new Error(
				`Engine "${this.config.engine}" does not implement embedding`,
			)
		}
		if (!request.input) {
			throw new Error('Input is required for embedding')
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: options?.timeout,
			signal: options?.signal,
		})
		taskLogger(LogLevels.verbose, 'Creating embedding task')
		const taskBegin = process.hrtime.bigint()
		const result = this.engine.processEmbeddingTask!(
			{
				request,
				config: this.config,
				log: taskLogger,
			},
			this.engineInstance,
			controller.signal,
		).then((result) => {
			const timeElapsed = elapsedMillis(taskBegin)
			controller.complete()
			if (controller.timeoutSignal.aborted) {
				taskLogger(LogLevels.warn, 'Embedding task timed out')
			}
			taskLogger(LogLevels.verbose, 'Embedding task done', {
				elapsed: timeElapsed,
			})
			return result
		}).catch((error) => {
			taskLogger(LogLevels.error, 'Task failed - ', {
				error,
			})
			throw error
		})

		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}

	processImageToTextTask(
		request: ImageToTextRequest,
		options?: ProcessingOptions,
	) {
		if (!('processImageToTextTask' in this.engine)) {
			throw new Error(
				`Engine "${this.config.engine}" does not implement image to text`,
			)
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: options?.timeout,
			signal: options?.signal,
		})
		const taskBegin = process.hrtime.bigint()
		const result = this.engine.processImageToTextTask!(
			{
				request,
				config: this.config,
				log: taskLogger,
			},
			this.engineInstance,
			controller.signal,
		).then((result) => {
			const timeElapsed = elapsedMillis(taskBegin)
			controller.complete()
			if (controller.timeoutSignal.aborted) {
				taskLogger(LogLevels.warn, 'ImageToText task timed out')
			}
			taskLogger(LogLevels.verbose, 'ImageToText task done', {
				elapsed: timeElapsed,
			})
			return result
		}).catch((error) => {
			taskLogger(LogLevels.error, 'Task failed - ', {
				error,
			})
			throw error
		})

		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}
	
	processSpeechToTextTask(
		request: SpeechToTextRequest,
		options?: SpeechToTextProcessingOptions,
	) {
		if (!('processSpeechToTextTask' in this.engine)) {
			throw new Error(
				`Engine "${this.config.engine}" does not implement speech to text`,
			)
		}
		this.lastUsed = Date.now()
		const id = this.generateTaskId()
		const taskLogger = withLogMeta(this.log, {
			sequence: this.currentRequest!.sequence,
			task: id,
		})
		const controller = this.createTaskController({
			timeout: options?.timeout,
			signal: options?.signal,
		})
		const taskBegin = process.hrtime.bigint()
		const result = this.engine.processSpeechToTextTask!(
			{
				request,
				config: this.config,
				log: taskLogger,
			},
			this.engineInstance,
			controller.signal,
		).then((result) => {
			const timeElapsed = elapsedMillis(taskBegin)
			controller.complete()
			if (controller.timeoutSignal.aborted) {
				taskLogger(LogLevels.warn, 'SpeechToText task timed out')
			}
			taskLogger(LogLevels.verbose, 'SpeechToText task done', {
				elapsed: timeElapsed,
			})
			return result
		}).catch((error) => {
			taskLogger(LogLevels.error, 'Task failed - ', {
				error,
			})
			throw error
		})

		return {
			id,
			model: this.modelId,
			createdAt: new Date(),
			cancel: controller.cancel,
			result,
		}
	}
}
