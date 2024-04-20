import os from 'node:os'
import path from 'node:path'
import { promises as fs, existsSync } from 'node:fs'
import PQueue from 'p-queue'
import { downloadFile } from 'ipull'
import * as gpt4all from './engines/gpt4all.js'
import * as nodeLlamaCpp from './engines/node-llama-cpp.js'

const engines = {
	gpt4all: gpt4all,
	'node-llama-cpp': nodeLlamaCpp,
}

export interface ChatMessage {
	role: 'system' | 'user' | 'assistant'
	content: string
}

export interface ChatCompletionRequest {
	model: string
	messages: ChatMessage[]
	cached?: boolean
}

type EngineType = 'gpt4all' | 'node-llama-cpp'

export interface LLMOptions {
	preload?: boolean
	gpu?: boolean | string
	url?: string
	file?: string
	engine?: EngineType
}

export interface LLMConfig extends Partial<LLMOptions> {
	name: string
	file: string
	engine: EngineType
}

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

type ChatCompletionResponse = any
type LLMEngineInstance = any
type CompletionToken = any

interface LLMEngine {
	createInstance: (config: LLMConfig) => Promise<LLMEngineInstance>
	disposeInstance: (instance: LLMEngineInstance) => Promise<void>
	onChatCompletionRequest: (
		instance: LLMEngineInstance,
		request: ChatCompletionRequest,
	) => ChatCompletionRequest
	processChatCompletion: (
		instance: LLMEngineInstance,
		request: ChatCompletionRequest,
		onToken?: (token: CompletionToken) => void,
	) => Promise<ChatCompletionResponse>
}

export class LLMInstance {
	locked: boolean
	model: string
	ref: LLMEngineInstance | null
	engine: LLMEngine
	config: LLMConfig
	download?: {
		progress: number
		speed: number
	}

	constructor(engine: LLMEngine, config: LLMConfig) {
		this.engine = engine
		this.config = config
		this.model = config.name
		this.ref = null
		this.locked = false
	}

	async init() {
		if (this.ref) {
			return
		}
		const time = Date.now()
		this.ref = await this.engine.createInstance(this.config)
		console.debug('Instance initialized', {
			model: this.model,
			time: Date.now() - time,
		})
	}

	dispose() {
		if (this.ref) {
			this.engine.disposeInstance(this.ref)
		}
	}

	lock() {
		this.locked = true
	}

	unlock() {
		this.locked = false
	}

	onChatCompletionRequest(request: ChatCompletionRequest) {
		console.debug('onChatCompletionRequest', request)
		return this.engine.onChatCompletionRequest(this.ref, request)
	}

	processChatCompletion(
		request: ChatCompletionRequest,
		onToken?: (token: CompletionToken) => void,
	) {
		console.debug('processChatCompletion', request)
		return this.engine.processChatCompletion(this.ref, request, onToken)
	}
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

export class LLMPool {
	queue: PQueue
	instances: Record<string, LLMInstance>
	config: LLMPoolConfig

	constructor(opts: LLMPoolOptions) {
		const modelsDir =
			opts.modelsDir || path.resolve(os.homedir(), '.cache/models')

		const config: LLMPoolConfig = {
			modelsDir,
			concurrency: 1,
			// ...opts,
			models: {},
		}

		// const modelsDir = opts.modelsDir || path.resolve(os.homedir(), '.cache')

		for (const modelName in opts.models) {
			const modelOpts = opts.models[modelName]
			const file = resolveModelFile(modelOpts, modelsDir)
			const modelConfig = {
				...modelOpts,
				file,
				name: modelName,
				engine: modelOpts.engine || 'gpt4all',
			}
			config.models[modelName] = modelConfig
		}

		this.queue = new PQueue({ concurrency: config.concurrency })
		this.config = config
		this.instances = {}
	}

	async dispose() {
		const disposePromises = Object.values(this.instances).map((instance) =>
			instance.dispose(),
		)
		await Promise.all(disposePromises)
	}

	// create model instances, preloading if configured
	async init() {
		await fs.mkdir(this.config.modelsDir, { recursive: true })
		const preloadPromises = []
		const modelConfigs = this.config.models
		
		for (const modelName in modelConfigs) {
			const config = modelConfigs[modelName]
			const engine = engines[config.engine]
			const key = `${modelName}`
			// const key = `${i}:${config.name}`
			this.instances[key] = new LLMInstance(engine, config)
			// this needs to happen beforehand, otherwise multiple instances 
			// (of the same model) might download the same file
			await this.prepareInstance(this.instances[key])
			if (config.preload) {
				preloadPromises.push(this.instances[key].init())
			}
		}

		await Promise.all(preloadPromises)
		console.debug('Pool initialized')
	}
	
	// prepare an instance, downloading the model weights if necessary
	async prepareInstance(instance: LLMInstance) {
		
		const config = instance.config
		if (!existsSync(config.file)) {
			if (config.url) {
				console.debug('download', {
					url: config.url,
					savePath: config.file,
				})
				const downloader = await downloadFile({
					url: config.url,
					savePath: config.file,
				})
				downloader.on('progress', (progress) => {
					instance.download = {
						progress: progress.percentage,
						speed: progress.speed,
					}
				})

				try {
					await downloader.download()
				} catch (error: any) {
					console.error(`Download failed: ${error.message}`)
				} finally {
					instance.download = undefined
				}
			}

			if (!existsSync(config.file)) {
				throw new Error(`Model file not found: ${config.file}`)
			}
		}
	}
	
	getStatusInfo() {
		return {
			queue: this.queue.size,
			instances: Object.fromEntries(
				Object.entries(this.instances).map(([key, instance]) => {
					return [
						key,
						{
							model: instance.model,
							locked: instance.locked,
							url: instance.config.url,
							file: instance.config.file,
							engine: instance.config.engine,
							download: instance.download,
						},
					]
				}),
			),
		}
	}

	// picks an instance from the pool to handle a chat completion request
	async requestChatCompletionInstance(request: ChatCompletionRequest) {
		console.debug('requestChatCompletionInstance', request)

		// 1st search for an instance that has the messages already ingested and the context ready
		for (const key in this.instances) {
			const instance = this.instances[key]
	
			if (instance.ref && !instance.locked && instance.model === request.model) {
				console.debug('hm', instance)
				// can't think of better names for either the left or right side of the assignment
				const preprocessedRequest = instance.onChatCompletionRequest(request)
				if (preprocessedRequest.cached) {
					console.debug('instance found by cached messages')
					instance.lock()
					return {
						instance,
						request: preprocessedRequest,
					}
				}
			}
		}

		// 2nd search for an instance that has the same model name
		const matchingInstances = []

		for (const key in this.instances) {
			const instance = this.instances[key]
			if (instance.model === request.model) {
				console.debug('instance found by model name')
				
				if (!instance.locked) {
					if (!instance.ref) {
						await instance.init()
					}
					instance.lock()
					return { instance, request }
				}
				matchingInstances.push(instance)
			}
		}

		// TODO no instance found for the requested model.
		// no matchingInstances -> error
		// matchingInstances -> wait for one to be available
		throw new Error('No available instances')
	}
}
