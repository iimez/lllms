import { promises as fs, existsSync, statSync } from 'node:fs'
import path from 'node:path'
import PQueue from 'p-queue'
import { ModelDownloader, createModelDownloader } from 'node-llama-cpp'
import { FormattedStatus } from 'ipull'
import { LLMConfig } from '#lllms/types/index.js'
import { GGUFMeta, readGGUFMetaFromFile } from '#lllms/lib/gguf.js'
import { calcFileChecksums } from '#lllms/lib/calcFileChecksums.js'
import { Logger, LogLevels, createLogger, LogLevel } from '#lllms/lib/logger.js'

interface DownloadTask {
	status: 'pending' | 'processing' | 'completed' | 'error'
	model: string
	handle: ModelDownloader
	progress: FormattedStatus | null
}

interface LLMMeta {
	gguf: Omit<GGUFMeta, 'tokenizer'>
	checksums?: Record<string, string>
}

export interface LLMStoreModelConfig extends LLMConfig {
	meta?: LLMMeta
}

export interface LLMMStoreOptions {
	modelsPath: string,
	models: Record<string, LLMConfig>,
	log?: Logger | LogLevel
}

export class LLMStore {
	downloadQueue: PQueue
	downloadTasks: DownloadTask[] = []
	modelsPath: string
	models: Record<string, LLMStoreModelConfig> = {}
	private log: Logger

	constructor(
		options: LLMMStoreOptions,
	) {
		if (options.log) {
			this.log = typeof options.log === 'string' ? createLogger(options.log) : options.log
		} else {
			this.log = createLogger(LogLevels.warn)
		}
		this.downloadQueue = new PQueue({
			concurrency: 1,
		})
		this.modelsPath = options.modelsPath
		this.models = options.models
	}

	async init() {
		if (!existsSync(this.modelsPath)) {
			await fs.mkdir(this.modelsPath, { recursive: true })
		}
		
		for (const modelId in this.models) {
			const model = this.models[modelId]
			if (model.prepare === 'blocking') {
				await this.prepareModel(modelId)
			} else if (model.prepare === 'async') {
				this.prepareModel(modelId)
			}
		}
	}

	getStatus() {
		const downloadStatus = Object.fromEntries(
			this.downloadTasks.map((task) => [
				task.model,
				{
					status: task.status,
					totalBytes: task.handle.totalSize,
					downloadedBytes: task.handle.downloadedSize,
					progress: task.progress ? {
						percentage: task.progress.percentage,
						speed: task.progress.speed,
						formattedSpeed: task.progress.formattedSpeed,
						timeLeft: task.progress.timeLeft,
						formatTimeLeft: task.progress.formatTimeLeft,
					} : null,
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

	// TODO pull out (file: string, checksums: Record<string, string>) => Promise<Meta>
	// any split metadata reading logic
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
		this.log(LogLevels.info, `Preparing model`, { model: modelId })
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
				const onCompleted = (task: DownloadTask) => {
					if (task.model === modelId) {
						if (task.status === 'error') {
							reject(new Error(`Download failed for ${modelId}`))
						} else {
							this.downloadQueue.off('completed', onCompleted)
							resolve()
						}
					}
				}
				this.downloadQueue.on('completed', onCompleted)
			})
		}

		this.log(LogLevels.info, `Downloading ${modelId}`, {
			url: model.url,
			file: model.file,
		})
		// otherwise, start a new download
		const task: DownloadTask = {
			model: modelId,
			status: 'pending',
			progress: null,
			handle: await createModelDownloader({
				modelUrl: model.url,
				dirPath: path.dirname(model.file),
				fileName: path.basename(model.file),
				deleteTempFileOnCancel: false,
				onProgress: (status) => {}, // this does not seem to be called
			}),
		}
		// @ts-ignore .. but setting onProgress will allow for this to be called
		task.handle._onDownloadProgress = (status: FormattedStatus) => {
			task.progress = status
		}
		
		const logInterval = setInterval(() => {
			if (!task.progress) {
				return
			}
			const percentage = task.progress.percentage.toFixed(2)
			const speed = task.progress.formattedSpeed
			const eta = task.progress.formatTimeLeft
			this.log(LogLevels.info, `Downloading ${modelId}: ${percentage}% at ${speed} - ETA: ${eta}`)
		}, 60000)

		if (signal) {
			signal.addEventListener('abort', () => {
				// task.handle.close()
			})
		}
		this.downloadTasks.push(task)
		try {
			await this.downloadQueue.add(async () => {
				task.status = 'processing'
				await task.handle.download({ signal })
				task.status = 'completed'
				return task
			})
		} catch (err) {
			task.status = 'error'
		} finally {
			this.downloadTasks = this.downloadTasks.filter((t) => t !== task)
			clearInterval(logInterval)
		}
	}
}
