import crypto from 'node:crypto'
import { existsSync } from 'node:fs'
import { customAlphabet } from 'nanoid'
import { EngineInstance } from './engines/index.js'
import {
	CompletionRequest,
	ChatCompletionRequest,
	LLMEngine,
	LLMConfig,
	EngineCompletionContext,
} from './types/index.js'
import { createContextStateHash } from './util/createContextStateHash.js'
import { LogLevels, Logger, createLogger } from './util/log.js'

const idAlphabet =
	'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
const generateId = customAlphabet(idAlphabet, 8)

type LLMInstanceStatus = 'idle' | 'busy' | 'error' | 'loading' | 'preparing'

interface LLMInstanceOptions extends LLMConfig {
	logger?: Logger
}

export class LLMInstance {
	id: string
	status: LLMInstanceStatus
	model: string
	config: LLMConfig
	fingerprint: string
	createdAt: Date
	lastUsed: number = 0
	logger: Logger

	private engine: LLMEngine
	private contextStateHash?: string
	private needsContextReset: boolean = false
	private llm: EngineInstance | null

	constructor(engine: LLMEngine, { logger, ...opts }: LLMInstanceOptions) {
		this.id = opts.name + ':' + generateId(8)
		this.engine = engine
		this.config = opts
		this.model = opts.name
		this.llm = null
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
		const time = Date.now()
		this.status = 'loading'

		this.llm = await this.engine.loadInstance(this.config, {
			instance: this.id,
			logger: this.logger,
			signal,
		})
		this.status = 'idle'

		this.logger(LogLevels.verbose, 'Instance loaded', {
			instance: this.id,
			elapsed: Date.now() - time,
		})
	}

	async dispose() {
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

	hasContext() {
		return this.contextStateHash !== undefined
	}

	matchesContext(req: CompletionRequest | ChatCompletionRequest) {
		if (!this.contextStateHash) {
			return false
		}
		const incomingStateHash = createContextStateHash(req, true)
		return this.contextStateHash === incomingStateHash
	}

	createChatCompletion(req: ChatCompletionRequest) {
		if (!req.messages) {
			throw new Error('Messages are required for chat completions.')
		}
		const id = this.id + '-' + generateId(8)
		this.lastUsed = Date.now()
		this.logger(LogLevels.verbose, 'Creating chat completion', { completion: id })
		return {
			id,
			model: this.model,
			createdAt: new Date(),
			process: async (ctx: Partial<EngineCompletionContext> = {}) => {
				const time = Date.now()
				if (this.needsContextReset) {
					this.contextStateHash = undefined
					this.needsContextReset = false
					ctx.resetContext = true
				}
				const result = await this.engine.processChatCompletion(this.llm, req, {
					...ctx,
					instance: this.id,
					logger: this.logger,
				})

				const newMessages = [...req.messages]
				newMessages.push(result.message)

				this.contextStateHash = createContextStateHash({
					systemPrompt: req.systemPrompt,
					messages: newMessages,
				})

				this.logger(LogLevels.verbose, 'Chat completion done', {
					completion: id,
					elapsed: Date.now() - time,
				})

				return result
			},
		}
	}

	createCompletion(completionArgs: CompletionRequest) {
		if (!completionArgs.prompt) {
			throw new Error('Prompt is required for completions.')
		}
		this.lastUsed = Date.now()
		const id = this.id + '-' + generateId(8)
		this.logger(LogLevels.verbose, 'Creating completion', { completion: id })

		return {
			id,
			model: this.model,
			createdAt: new Date(),
			process: async (ctx: Partial<EngineCompletionContext> = {}) => {
				const time = Date.now()
				const result = await this.engine.processCompletion(
					this.llm,
					completionArgs,
					{
						...ctx,
						logger: this.logger,
						instance: this.id,
					},
				)
				this.logger(LogLevels.verbose, 'Completion done', {
					completion: id,
					elapsed: Date.now() - time,
				})
				return result
			},
		}
	}
}
