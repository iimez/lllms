import os from 'node:os'
import path from 'node:path'
import crypto from 'node:crypto'
import { promises as fs, existsSync } from 'node:fs'
import PQueue from 'p-queue'
import { downloadFile } from 'ipull'
import { engines, EngineType } from './engines/index.js'
import { ChatMessage, ChatCompletionArgs, EngineChatCompletionResult } from './types/index.js'
import { nanoid } from 'nanoid'
export interface LLMOptions {
	gpu?: boolean | string
	url?: string
	file?: string
	engine: EngineType
	minInstances?: number
	maxInstances?: number
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

interface TaskResult {
	instance: LLMInstance
	args: any
}

type LLMEngineInstance = any

interface LLMEngine {
	loadInstance: (config: LLMConfig) => Promise<LLMEngineInstance>
	disposeInstance: (instance: LLMEngineInstance) => Promise<void>
	processChatCompletion: (
		instance: LLMEngineInstance,
		args: ChatCompletionArgs,
	) => Promise<EngineChatCompletionResult>
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

interface ChatState {
	messages: ChatMessage[]
	systemPrompt?: string
}
function createChatStateHash(
	state: ChatState,
	dropLastMessage: boolean = false,
): string {
	const data = {
		messages: [...state.messages],
		systemPrompt: state.systemPrompt,
	}
	if (dropLastMessage && data.messages.length > 1) {
		data.messages.pop()
	}
	return crypto.createHash('sha1').update(JSON.stringify(data)).digest('hex')
}

export class LLMInstance {
	id: string
	locked: boolean
	model: string
	llm: LLMEngineInstance | null
	engine: LLMEngine
	config: LLMConfig
	download?: {
		progress: number
		speed: number
	}
	chatStateHash?: string
	fingerprint: string
        created_by: Date
	constructor(engine: LLMEngine, config: LLMConfig) {
		this.id = config.name + ':' + nanoid()
		this.engine = engine
		this.config = config
		this.model = config.name
		this.llm = null
		this.locked = false
		// TODO to implement this properly we should only include what changes the "behavior" of the model
		this.fingerprint = crypto.createHash('sha1').update(JSON.stringify(config)).digest('hex')
                this.created_by = new Date()
	}

	async load() {
		if (this.llm) {
			return
		}
		const time = Date.now()
		this.llm = await this.engine.loadInstance(this.config)
		console.debug('Instance initialized', {
			id: this.id,
			time: Date.now() - time,
		})
	}

	dispose() {
		if (this.llm) {
			this.engine.disposeInstance(this.llm)
		}
	}

	lock() {
		this.locked = true
	}

	unlock() {
		this.locked = false
	}

	async processChatCompletion(args: ChatCompletionArgs) {
		console.debug('processChatCompletion', args)
		const result = await this.engine.processChatCompletion(this.llm, args)
		console.debug('processChatCompletion result', result)
		return {
			id: this.config.engine + '-' + nanoid(),
			model: this.model,
			created: new Date(),
			...result,
		}
	}
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

		for (const modelName in opts.models) {
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
	}

	async dispose() {
		const disposePromises = Object.values(this.instances).map((instance) =>
			instance.dispose(),
		)
		await Promise.all(disposePromises)
	}

	// initializing pool, creating instances and loading models
	async init() {
		await fs.mkdir(this.config.modelsDir, { recursive: true })
		const loadingPromises = []
		const modelConfigs = this.config.models

		for (const modelName in modelConfigs) {
			const modelConfig = modelConfigs[modelName]
			const engineMethods = engines[modelConfig.engine]
			const instanceCount = modelConfig.minInstances ?? 0

			for (let i = 0; i < instanceCount; i++) {
				const instance = new LLMInstance(engineMethods, modelConfig)
				this.instances[instance.id] = instance
				// TODO rethink model download strategy, consider doing this outside of pool init
				// - models could be downloaded in parallel, might be faster in total?
				// - or sequentially, to make them available as fast as possible
				// currently we download sequentially, but we still wait until all are done to continue
				await this.downloadInstanceModel(instance)
				loadingPromises.push(instance.load())
			}
		}

		await Promise.all(loadingPromises)
		// console.debug('Pool initialized')
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

	// download weights for a model, if necessary
	private async downloadInstanceModel(instance: LLMInstance) {
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

	private canSpawnInstance(modelName: string) {
		const modelConfig = this.config.models[modelName]
		const maxInstances = modelConfig.maxInstances ?? 1
		const currentInstances = Object.values(this.instances).filter(
			(instance) => instance.model === modelName,
		)
		return currentInstances.length < maxInstances
	}

	// acquire an instance from the pool, or create a new one if necessary
	// the instance will be locked and must be released when done
	async acquireInstance(modelName: string) {
		
		// first check if theres an available instance that can be reused
		for (const key in this.instances) {
			const instance = this.instances[key]
			if (!instance.locked && instance.model === modelName) {
				console.debug('using loaded instance')
				// TODO make sure we reset context
				instance.lock()
				return instance
			}
		}

		// see if we can spawn a new instance
		if (this.canSpawnInstance(modelName)) {
			console.debug('spawning new instance')
			const modelConfig = this.config.models[modelName]
			const engineMethods = engines[modelConfig.engine]
			const instance = new LLMInstance(engineMethods, modelConfig)
			this.instances[instance.id] = instance
			await this.downloadInstanceModel(instance)
			await instance.load()
			instance.lock()
			return instance
		}

		// wait until an instance of that model is released
		const modelInstanceIsReleased = (): Promise<LLMInstance> => {
			return new Promise((resolve, reject) => {
				const onComplete = ({ instance, args }: TaskResult) => {
					if (instance.model === args.model) {
						this.queue.off('completed', onComplete)
						instance.lock()
						resolve(instance)
					}
				}
				this.queue.on('completed', onComplete)
			})
		}

		return await modelInstanceIsReleased()
	}

	// attepts to acquire an instance that has the chat state ready, or forwards to acquireInstance
	async acquireChatCompletionInstance(args: ChatCompletionArgs) {
		// for chat completions first search for an instance that has the messages already ingested and the context ready
		const incomingChatHash = createChatStateHash(args, true)
		for (const key in this.instances) {
			const instance = this.instances[key]

			if (
				!instance.locked &&
				instance.model === args.model &&
				instance.chatStateHash === incomingChatHash
			) {
				console.debug('using cached instance', instance)
				// return createInstanceLock(instance)
				instance.lock()
				return instance
			}
		}

		return await this.acquireInstance(args.model)
	}

	// requests an instance from the pool to handle a chat completion request
	async requestChatCompletionInstance(args: ChatCompletionArgs) {
		if (!this.config.models[args.model]) {
			throw new Error(`Model not found: ${args.model}`)
		}
		console.debug('requesting chat completion instance', args)

		const instance = await this.acquireChatCompletionInstance(args)
		
		// once instance is acquired & locked, we can pass it on to the caller
		// the queue task promise will be forwarded as "relase" function
		let resolveTask: (value: TaskResult) => void

		this.queue.add(() => {
			return new Promise((resolve, reject) => {
				resolveTask = resolve
			})
		})

		return {
			instance,
			release: () => {
				instance.unlock()
				resolveTask({ instance, args })
			},
		}
	}
}
