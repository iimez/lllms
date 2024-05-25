import { promises as fs, existsSync, statSync } from 'node:fs'
import PQueue from 'p-queue'
import { downloadFile, DownloadEngineNodejs } from 'ipull'
import { LLMConfig } from '#lllms/types/index.js'
import { GGUFMeta, readGGUFMetaFromFile } from '#lllms/lib/gguf.js'
import { calcFileChecksums } from '#lllms/lib/calcFileChecksums.js'
import { Logger, LogLevels, createLogger, LogLevel } from '#lllms/lib/logger.js'

interface DownloadTask {
	status: 'pending' | 'processing' | 'completed' | 'error'
	model: string
	handle: DownloadEngineNodejs
}

interface LLMMeta {
	gguf: Omit<GGUFMeta, 'tokenizer'>
	checksums?: Record<string, string>
}

export interface LLMStoreModelConfig extends LLMConfig {
	meta?: LLMMeta
}

export interface LLMMStoreOptions {
	maxDownloads?: number
	modelsPath: string,
	models: Record<string, LLMConfig>,
	log?: Logger | LogLevel
}

export class LLMStore {
	downloadQueue: PQueue
	downloadTasks: DownloadTask[] = []
	modelsPath: string
	models: Record<string, LLMStoreModelConfig> = {}
	private logger: Logger

	constructor(
		options: LLMMStoreOptions,
	) {
		if (options.log) {
			this.logger = typeof options.log === 'string' ? createLogger(options.log) : options.log
		} else {
			this.logger = createLogger(LogLevels.warn)
		}
		this.downloadQueue = new PQueue({
			concurrency: options.maxDownloads ?? 1,
		})
		this.modelsPath = options.modelsPath
		this.models = options.models
	}

	async init() {
		if (!existsSync(this.modelsPath)) {
			await fs.mkdir(this.modelsPath, { recursive: true })
		}
	}

	getStatus() {
		const downloadStatus = Object.fromEntries(
			this.downloadTasks.map((task) => [
				task.model,
				{
					status: task.status,
					progress: {
						percentage: task.handle.status.percentage,
						speed: task.handle.status.speed,
						speedFormatted: task.handle.status.formattedSpeed,
						timeLeft: task.handle.status.timeLeft,
						timeLeftFormatted: task.handle.status.formatTimeLeft,
					},
				},
			]),
		)

		const modelStatus = Object.fromEntries(
			Object.entries(this.models).map(([modelId, model]) => {
				let fileSize = 0
				let lastModified = ''
				if (existsSync(model.file)) {
					const stat = statSync(model.file)
					fileSize = stat.size
					lastModified = new Date(stat.mtimeMs).toISOString()
				}
				return [
					modelId,
					{
						engine: model.engine,
						engineOptions: model.engineOptions,
						minInstances: model.minInstances,
						maxInstances: model.maxInstances,
						source: {
							url: model.url,
							file: model.file,
							size: fileSize,
							lastModified,
							checksums: model.meta?.checksums,
							gguf: model.meta?.gguf,
							download: downloadStatus[modelId],
						},
					},
				]
			}),
		)

		return modelStatus
	}

	async validateModel(modelId: string) {
		const model = this.models[modelId]
		const ggufMeta = await readGGUFMetaFromFile(model.file)

		if (model.md5) {
			const fileChecksums = await calcFileChecksums(model.file, ['md5'])
			if (fileChecksums.md5 !== model.md5) {
				throw new Error(`MD5 checksum mismatch for ${modelId} - expected ${model.md5} got ${fileChecksums.md5}`)
			}
		}

		if (model.sha256) {
			const fileChecksums = await calcFileChecksums(model.file, ['sha256'])
			if (fileChecksums.sha256 !== model.sha256) {
				throw new Error(`SHA256 checksum mismatch for ${modelId} - expected ${model.sha256} got ${fileChecksums.sha256}`)
			}
		}

		let checksums: Record<string, string> | undefined
		const calcChecksums = false
		if (calcChecksums) {
			if (!model.sha256 && !model.md5) {
				checksums = await calcFileChecksums(model.file, ['md5', 'sha256'])
			} else {
				checksums = {}
				if (model.md5) {
					checksums.md5 = model.md5
				}
				if (model.sha256) {
					checksums.sha256 = model.sha256
				}
			}
		}
		
		const gguf = ggufMeta
		if (gguf.tokenizer?.ggml) {
			delete gguf.tokenizer.ggml.merges
			delete gguf.tokenizer.ggml.tokens
			delete gguf.tokenizer.ggml.scores
			delete gguf.tokenizer.ggml.token_type
		}

		return {
			gguf,
			checksums,
		}
	}

	async prepareModel(modelId: string, signal?: AbortSignal) {
		const model = this.models[modelId]
		// make sure the model files exists, download if possible.
		if (!existsSync(model.file) && model.url) {
			await this.downloadModel(modelId, signal)
		}
		if (!existsSync(model.file)) {
			throw new Error(`Model file not found: ${model.file}`)
		}
		// read some model metadata and verify checksums
		if (!model.meta) { // TODO could more explicitly signify "this model/file has already been validated"
			model.meta = await this.validateModel(modelId)
		}
	}

	async downloadModel(modelId: string, signal?: AbortSignal) {
		const model = this.models[modelId]

		if (!model.url) {
			throw new Error(`Model ${modelId} does not have a URL`)
		}
		// if the model is already being downloaded, wait for it to complete
		const existingTask = this.downloadTasks.find((t) => t.model === modelId)
		if (existingTask) {
			return new Promise<void>((resolve, reject) => {
				existingTask.handle.once('completed', () => resolve())
				existingTask.handle.once('error', (error) => reject(error))
			})
		}

		this.logger(LogLevels.info, `Downloading ${modelId}`, {
			url: model.url,
			file: model.file,
		})
		// otherwise, start a new download
		const task: DownloadTask = {
			model: modelId,
			status: 'pending',
			handle: await downloadFile({
				url: model.url,
				savePath: model.file,
				// parallelStreams: 3 // Number of parallel connections (default: 3)
			}),
		}
		
		const logInterval = setInterval(() => {
			const percentage = task.handle.status.percentage.toFixed(2)
			const speed = task.handle.status.formattedSpeed
			const eta = task.handle.status.formatTimeLeft
			this.logger(LogLevels.info, `Downloading ${modelId}: ${percentage}% at ${speed} - ETA: ${eta}`)
		}, 60000)

		if (signal) {
			signal.addEventListener('abort', () => {
				task.handle.close()
			})
		}
		this.downloadTasks.push(task)
		try {
			await this.downloadQueue.add(() => {
				task.status = 'processing'
				return task.handle.download()
			})
			task.status = 'completed'
		} finally {
			this.downloadTasks = this.downloadTasks.filter((t) => t !== task)
			clearInterval(logInterval)
		}
	}
}
