import os from 'node:os'
import path from 'node:path'
import crypto from 'node:crypto'
import { promises as fs, existsSync } from 'node:fs'
import { nanoid } from 'nanoid'
import PQueue from 'p-queue'
import { downloadFile } from 'ipull'
import { engines, EngineInstance } from './engines/index.js'
import {
	ChatMessage,
	CompletionRequest,
	GenerationArgs,
	LLMEngine,
	LLMConfig,
	LLMOptions,
} from './types/index.js'
import type { ProgressStatusWithIndex } from 'ipull/dist/download/transfer-visualize/progress-statistics-builder.js'

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

interface DownloadOptions {
	downloadUrl?: string
	signal?: AbortSignal
	onProgress?: (progress: ProgressStatusWithIndex) => void
	onError?: (error: Error) => void
}

async function ensureModelFile(file: string, opts: DownloadOptions = {}) {
	// const config = instance.config
	if (!existsSync(file)) {
		if (opts.downloadUrl) {
			console.debug('download', opts)
			const downloader = await downloadFile({
				url: opts.downloadUrl,
				savePath: file,
				// parallelStreams: 3 // Number of parallel connections (default: 3)
			})
			if (opts.signal) {
				opts.signal.addEventListener('abort', () => {
					downloader.close()
				})
			}
			if (opts.onProgress) {
				downloader.on('progress', opts.onProgress)
			}

			try {
				await downloader.download()
			} catch (error: any) {
				console.error(`Download failed: ${error.message}`)
				if (opts.onError) {
					opts.onError(error)
				}
			}
		}

		if (!existsSync(file)) {
			throw new Error(`Failed to assert model file at ${file}`)
		}
	}
}

interface ContextStateData {
	messages?: ChatMessage[]
	prompt?: string
	systemPrompt?: string
}
function createContextStateHash(
	state: ContextStateData,
	dropLastMessage: boolean = false,
): string {
	const messages = state.messages ? [...state.messages] : []
	if (dropLastMessage && messages.length > 1) {
		messages.pop()
	}
	return crypto
		.createHash('sha1')
		.update(
			JSON.stringify({
				messages,
				prompt: state.prompt,
				systemPrompt: state.systemPrompt,
			}),
		)
		.digest('hex')
}

class AbortError extends Error {
	constructor(message = 'Operation aborted') {
		super(message)
		this.name = 'AbortError'
	}
}

export class LLMInstance {
	id: string
	locked: boolean
	model: string
	llm: EngineInstance | null
	engine: LLMEngine
	config: LLMConfig
	download?: {
		progress: number
		speed: number
	}
	contextStateHash?: string
	fingerprint: string
        createdBy: Date
	constructor(engine: LLMEngine, config: LLMConfig) {
		this.id = config.name + ':' + nanoid()
		this.engine = engine
		this.config = config
		this.model = config.name
		this.llm = null
		this.locked = false
		// TODO to implement this properly we should only include what changes the "behavior" of the model
                this.createdBy = new Date()
		this.fingerprint = crypto
			.createHash('sha1')
			.update(JSON.stringify(config))
			.digest('hex')
	}

	async load(signal?: AbortSignal) {
		if (this.llm) {
			return
		}
		const time = Date.now()
		this.llm = await this.engine.loadInstance(this.config, signal)
		console.debug('Instance initialized', {
			id: this.id,
			tookMillis: Date.now() - time,
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

	createChatCompletion(completionArgs: CompletionRequest) {
		const id = this.config.engine + '-' + nanoid()

		console.debug(
			'Creating ChatCompletion',
			id,
			JSON.stringify(completionArgs, null, 2),
		)
		// const created
		// const result = await this.engine.processChatCompletion(this.llm, args)
		// console.debug('processChatCompletion result', result)
		return {
			id,
			model: this.model,
			createdAt: new Date(),
			process: async (processingArgs: GenerationArgs) => {
				const time = Date.now()
				const result = await this.engine.processChatCompletion(
					this.llm,
					completionArgs,
					processingArgs,
				)
				console.debug('Finished ChatCompletion', {
					id,
					completion: result,
					tookMillis: Date.now() - time,
				})
				return result
			},
		}
	}

	createCompletion(completionArgs: CompletionRequest) {
		const id = this.config.engine + '-' + nanoid()

		console.debug('Creating Completion', {
			id,
			args: completionArgs,
		})

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
				console.debug('Finished ChatCompletion', {
					id,
					completion: result,
					tookMillis: Date.now() - time,
				})
				return result
			},
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
				await ensureModelFile(modelConfig.file, {
					downloadUrl: modelConfig.url,
					onProgress: (progress) => {
						instance.download = {
							progress: progress.percentage,
							speed: progress.speed,
						}
					},
				})
				instance.download = undefined
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
	async acquireInstance(modelName: string, signal?: AbortSignal) {
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
			// await this.downloadInstanceModel(instance)
			await ensureModelFile(modelConfig.file, {
				downloadUrl: modelConfig.url,
				signal,
				onProgress: (progress) => {
					instance.download = {
						progress: progress.percentage,
						speed: progress.speed,
					}
				},
			})
			await instance.load(signal)
			instance.lock()
			return instance
		}

		// wait until an instance of that model is released
		const modelInstanceReleased = (): Promise<LLMInstance> => {
			return new Promise((resolve, reject) => {
				const onComplete = ({ instance, args }: TaskResult) => {
					console.debug('task result is', { instance, args })
					if (instance.model === args.model) {
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

		return await modelInstanceReleased()
	}

	// attepts to acquire an instance that has the chat state ready, or forwards to acquireInstance
	async acquireCachedInstance(args: CompletionRequest, signal?: AbortSignal) {
		// for completions first search for an instance that has the messages already ingested and the context ready
		const incomingStateHash = createContextStateHash(args, true)
		for (const key in this.instances) {
			const instance = this.instances[key]

			if (
				!instance.locked &&
				instance.model === args.model &&
				instance.contextStateHash === incomingStateHash
			) {
				console.debug('reusing cached instance', instance)
				if (signal?.aborted) {
					throw new AbortError('Processing cancelled')
				}
				instance.lock()
				return instance
			}
		}

		return await this.acquireInstance(args.model, signal)
	}

	// requests an instance from the pool to handle a chat completion request
	async requestCompletionInstance(
		args: CompletionRequest,
		signal: AbortSignal,
	) {
		if (!this.config.models[args.model]) {
			throw new Error(`Model not found: ${args.model}`)
		}
		console.debug('requesting chat completion instance', args)

		const instance = await this.acquireCachedInstance(args, signal)

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
