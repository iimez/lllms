import crypto from 'node:crypto'
import { existsSync } from 'node:fs'
import { customAlphabet } from 'nanoid'
import {
	CompletionRequest,
	ChatCompletionRequest,
	LLMEngine,
	LLMConfig,
	LLMRequest,
	CompletionProcessingOptions,
} from '#lllms/types/index.js'
import type { EngineInstance } from '#lllms/engines/index.js'
import { createContextStateHash } from '#lllms/lib/createContextStateHash.js'
import { LogLevels, Logger, createLogger, withLogMeta } from '#lllms/lib/logger.js'
import { elapsedMillis, mergeAbortSignals } from '#lllms/lib/util.js'

const idAlphabet =
	'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
const generateId = customAlphabet(idAlphabet, 8)

type LLMInstanceStatus = 'idle' | 'busy' | 'error' | 'loading' | 'preparing'

interface LLMInstanceOptions extends LLMConfig {
	logger?: Logger
	gpu: boolean
}

export class LLMInstance {
	id: string
	status: LLMInstanceStatus
	model: string
	config: LLMConfig
	fingerprint: string
	createdAt: Date
	lastUsed: number = 0
	gpu: boolean
	ttl: number
	logger: Logger

	private engine: LLMEngine
	private contextStateHash?: string
	private needsContextReset: boolean = false
	private llm: EngineInstance | null = null
	private currentRequest: LLMRequest | null = null

	constructor(engine: LLMEngine, { logger, gpu, ...opts }: LLMInstanceOptions) {
		this.id = opts.id + ':' + generateId(8)
		this.engine = engine
		this.config = opts
		this.model = opts.id
		this.gpu = gpu
		this.ttl = opts.ttl ?? 300
		this.status = 'preparing'
		this.createdAt = new Date()
		this.logger = withLogMeta(logger ?? createLogger(LogLevels.warn), {
			instance: this.id,
		})

		// TODO to implement this properly we should only include what changes the "behavior" of the model
		this.fingerprint = crypto
			.createHash('sha1')
			.update(JSON.stringify(opts))
			.digest('hex')
		this.logger(LogLevels.info, 'Initializing new instance for', {
			model: this.model,
			gpu: this.gpu,
		})
	}

	async load(signal?: AbortSignal) {
		if (this.llm) {
			return
		}

		if (!existsSync(this.config.file)) {
			throw new Error(`Model file not found: ${this.config.file}`)
		}
		this.status = 'loading'
		const loadBegin = process.hrtime.bigint()
		this.llm = await this.engine.loadInstance(
			{
				log: withLogMeta(this.logger, {
					instance: this.id,
				}),
				config: {
					...this.config,
					engineOptions: {
						...this.config.engineOptions,
						gpu: this.gpu,
					},
				},
			},
			signal,
		)
		this.status = 'idle'
		this.logger(LogLevels.debug, 'Instance loaded', {
			elapsed: elapsedMillis(loadBegin),
		})
	}

	async dispose() {
		this.status = 'busy'
		if (this.llm) {
			await this.engine.disposeInstance(this.llm)
		}
	}

	lock(request: LLMRequest) {
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

	getContextState() {
		return this.contextStateHash
	}

	hasContextState() {
		return this.contextStateHash !== undefined
	}

	matchesContextState(request: LLMRequest) {
		if (!this.contextStateHash) {
			return false
		}
		const incomingStateHash = createContextStateHash(request, true)
		return this.contextStateHash === incomingStateHash
	}
	
	matchesRequirements(request: LLMRequest) {
		const mustGpu = this.config.engineOptions?.gpu === true
		const modelMatches = this.model === request.model
		const gpuMatches = mustGpu ? this.gpu : true
		return modelMatches && gpuMatches
	}
	
	createChatCompletion(request: ChatCompletionRequest) {
		if (!request.messages) {
			throw new Error('Messages are required for chat completions.')
		}
		const id = this.id + '-' + generateId(8)
		this.lastUsed = Date.now()
		const completionLogger = withLogMeta(this.logger, {
			sequence: this.currentRequest!.sequence,
			completion: id,
		})
		completionLogger(LogLevels.verbose, 'Creating chat completion')
		const cancelController = new AbortController()
		const cancel = () => {
			cancelController.abort()
		}

		return {
			id,
			model: this.model,
			createdAt: new Date(),
			cancel,
			process: async (opts?: CompletionProcessingOptions) => {
				// setting up signals
				const abortSignals = [
					cancelController.signal,
				]
				const timeoutController = new AbortController()
				let timeout: NodeJS.Timeout | undefined
				if (opts?.timeout) {
					timeout = setTimeout(() => {
						timeoutController.abort()
					}, opts?.timeout)
					abortSignals.push(timeoutController.signal)
				}
				if (opts?.signal) {
					abortSignals.push(opts?.signal)
				}
				// checking if this instance has been flagged for reset
				let resetContext = false
				if (this.needsContextReset) {
					this.contextStateHash = undefined
					this.needsContextReset = false
					resetContext = true
				}
				completionLogger(LogLevels.info, 'Processing chat completion', {
					resetContext,
				})
				const processingBegin = process.hrtime.bigint()
				const result = await this.engine.processChatCompletion(
					this.llm,
					{
						request,
						resetContext,
						config: this.config,
						log: completionLogger,
						onChunk: opts?.onChunk,
					},
					mergeAbortSignals(abortSignals),
				)
				const processingElapsed = elapsedMillis(processingBegin)

				if (timeout) {
					clearTimeout(timeout)
					if (timeoutController.signal.aborted) {
						completionLogger(LogLevels.warn, 'Chat completion timed out')
						result.finishReason = 'timeout'
					}
				}
				const newMessages = [...request.messages]
				newMessages.push(result.message)

				this.contextStateHash = createContextStateHash({
					systemPrompt: request.systemPrompt,
					messages: newMessages,
				})

				completionLogger(LogLevels.info, 'Chat completion done', {
					elapsed: processingElapsed,
				})

				return result
			},
		}
	}

	createCompletion(request: CompletionRequest) {
		if (!request.prompt) {
			throw new Error('Prompt is required for completions.')
		}
		this.lastUsed = Date.now()
		const id = this.id + '-' + generateId(8)
		const completionLogger = withLogMeta(this.logger, {
			sequence: this.currentRequest!.sequence,
			completion: id,
		})
		completionLogger(LogLevels.verbose, 'Creating chat completion')
		const cancelController = new AbortController()
		const cancel = () => {
			cancelController.abort()
		}
		return {
			id,
			model: this.model,
			createdAt: new Date(),
			cancel,
			process: async (opts?: CompletionProcessingOptions) => {
				const processingBegin = process.hrtime.bigint()
				const result = await this.engine.processCompletion(
					this.llm,
					{
						request,
						config: this.config,
						log: completionLogger,
						onChunk: opts?.onChunk,
					},
					opts?.signal,
				)
				const processingElapsed = elapsedMillis(processingBegin)
				this.contextStateHash = createContextStateHash({
					prompt: request.prompt,
				})
				completionLogger(LogLevels.verbose, 'Completion done', {
					elapsed: processingElapsed,
				})
				return result
			},
		}
	}
}
