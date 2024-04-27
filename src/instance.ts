import crypto from 'node:crypto'
import { existsSync } from 'node:fs'
import { EngineInstance } from './engines/index.js'
import {
	CompletionRequest,
	ChatCompletionRequest,
	GenerationArgs,
	LLMEngine,
	LLMConfig,
} from './types/index.js'
import { createContextStateHash, generateId } from './util/index.js'

type LLMInstanceStatus = 'idle' | 'busy' | 'error' | 'loading' | 'preparing'

export class LLMInstance {
	id: string
	status: LLMInstanceStatus
	model: string
	llm: EngineInstance | null
	engine: LLMEngine
	config: LLMConfig
	contextStateHash?: string
	fingerprint: string
	createdAt: Date
	lastUse: number = 0
	modelCreatedAt?: Date
	needsContextReset: boolean = false
	constructor(engine: LLMEngine, config: LLMConfig) {
		this.id = config.name + ':' + generateId(8)
		this.engine = engine
		this.config = config
		this.model = config.name
		this.llm = null
		this.status = 'preparing'
		this.createdAt = new Date()

		// TODO to implement this properly we should only include what changes the "behavior" of the model
		this.fingerprint = crypto
			.createHash('sha1')
			.update(JSON.stringify(config))
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

		this.llm = await this.engine.loadInstance(this.config, signal)
		this.status = 'idle'

		console.debug(`${this.id} instance loaded in ${Date.now() - time}ms`)
	}

	dispose() {
		if (this.llm) {
			this.engine.disposeInstance(this.llm)
		}
	}

	lock() {
		this.status = 'busy'
	}

	unlock() {
		this.status = 'idle'
	}

	resetContext() {
		this.needsContextReset = true
	}

	createChatCompletion(completionArgs: ChatCompletionRequest) {
		if (!completionArgs.messages) {
			throw new Error('Messages are required for chat completions.')
		}
		this.lastUse = Date.now()
		const id = this.id + '-' + generateId(8)
		console.debug(`${id} creating chat completion`)
		return {
			id,
			model: this.model,
			createdAt: new Date(),
			process: async (processingArgs: GenerationArgs = {}) => {
				const time = Date.now()
				if (this.needsContextReset) {
					this.contextStateHash = undefined
					this.needsContextReset = false
					processingArgs.resetContext = true
				}
				const result = await this.engine.processChatCompletion(
					this.llm,
					completionArgs,
					processingArgs,
				)

				const newMessages = [...completionArgs.messages]
				newMessages.push(result.message)

				this.contextStateHash = createContextStateHash({
					systemPrompt: completionArgs.systemPrompt,
					messages: newMessages,
					prompt: undefined,
				})

				console.debug(
					`${id} chat completion done in ${Date.now() - time}ms`,
					result.message,
				)
				return result
			},
		}
	}

	createCompletion(completionArgs: CompletionRequest) {
		if (!completionArgs.prompt) {
			throw new Error('Prompt is required for completions.')
		}
		const id = this.id + '-' + generateId(8)
		console.debug(`${id} creating completion`)

		return {
			id,
			model: this.model,
			createdAt: new Date(),
			process: async (processingArgs: GenerationArgs) => {
				const time = Date.now()
				const result = await this.engine.processCompletion(
					this.llm,
					completionArgs,
					processingArgs,
				)
				console.debug(`${id} completion done in ${Date.now() - time}ms`)
				return result
			},
		}
	}
}
