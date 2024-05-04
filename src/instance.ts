import crypto from 'node:crypto'
import { existsSync } from 'node:fs'
import { customAlphabet } from 'nanoid'
import { EngineInstance } from './engines/index.js'
import {
	CompletionRequest,
	ChatCompletionRequest,
	LLMEngine,
	LLMConfig,
	LLMRequest,
	CompletionProcessingOptions,
} from '#lllms/types/index.js'
import { createContextStateHash } from '#lllms/lib/createContextStateHash.js'
import { LogLevels, Logger, createLogger } from '#lllms/lib/logger.js'
import { elapsedMillis } from '#lllms/lib/elapsedMillis.js'

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
	private llm: EngineInstance | null

	constructor(engine: LLMEngine, { logger, gpu, ...opts }: LLMInstanceOptions) {
		this.id = opts.name + ':' + generateId(8)
		this.engine = engine
		this.config = opts
		this.model = opts.name
		this.llm = null
		this.gpu = gpu
		this.ttl = opts.ttl ?? 300
		this.status = 'preparing'
		this.createdAt = new Date()
		this.logger = logger ?? createLogger(LogLevels.warn)

		// TODO to implement this properly we should only include what changes the "behavior" of the model
		this.fingerprint = crypto
			.createHash('sha1')
			.update(JSON.stringify(opts))
			.digest('hex')
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
				id: this.id,
				log: this.logger,
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
			instance: this.id,
			elapsed: elapsedMillis(loadBegin),
		})
	}

	async dispose() {
		this.status = 'busy'
		if (this.llm) {
			await this.engine.disposeInstance(this.llm)
		}
	}

	lock() {
		if (this.status !== 'idle') {
			throw new Error(`Cannot lock: Instance ${this.id} is not idle`)
		}
		this.status = 'busy'
	}

	unlock() {
		this.status = 'idle'
	}

	resetContext() {
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
		this.logger(LogLevels.verbose, 'Creating chat completion', {
			completion: id,
		})
		return {
			id,
			model: this.model,
			createdAt: new Date(),
			process: async (opts?: CompletionProcessingOptions) => {
				let resetContext = false
				if (this.needsContextReset) {
					this.contextStateHash = undefined
					this.needsContextReset = false
					resetContext = true
				}
				const processBegin = process.hrtime.bigint()
				const result = await this.engine.processChatCompletion(
					this.llm,
					{
						request,
						config: this.config,
						id: this.id,
						log: this.logger,
						resetContext,
						onChunk: opts?.onChunk,
					},
					opts?.signal,
				)
				const processElapsed = elapsedMillis(processBegin)
				const newMessages = [...request.messages]
				newMessages.push(result.message)

				this.contextStateHash = createContextStateHash({
					systemPrompt: request.systemPrompt,
					messages: newMessages,
				})

				this.logger(LogLevels.info, 'Chat completion done', {
					completion: id,
					elapsed: processElapsed,
				})

				return result
			},
		}
	}

	createCompletion(req: CompletionRequest) {
		if (!req.prompt) {
			throw new Error('Prompt is required for completions.')
		}
		this.lastUsed = Date.now()
		const id = this.id + '-' + generateId(8)
		this.logger(LogLevels.verbose, 'Creating completion', { completion: id })

		return {
			id,
			model: this.model,
			createdAt: new Date(),
			process: async (opts?: CompletionProcessingOptions) => {
				const processBegin = process.hrtime.bigint()
				const result = await this.engine.processCompletion(
					this.llm,
					{
						config: this.config,
						log: this.logger,
						id: this.id,
						request: req,
						onChunk: opts?.onChunk,
					},
					opts?.signal,
				)
				
				this.logger(LogLevels.verbose, 'Completion done', {
					completion: id,
					elapsed: elapsedMillis(processBegin),
				})
				return result
			},
		}
	}
}
