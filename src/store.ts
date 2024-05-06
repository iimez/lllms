import path from 'node:path'
import { promises as fs, existsSync, statSync } from 'node:fs'
import PQueue from 'p-queue'
import { downloadFile, DownloadEngineNodejs } from 'ipull'
import { LLMConfig, LLMOptions } from '#lllms/types/index.js'
import { GGUFMeta, readGGUFMetaFromFile } from '#lllms/lib/gguf.js'
import { calcFileChecksums } from '#lllms/lib/calcFileChecksums.js'

const modelIdPattern = /^[a-zA-Z0-9_:\-]+$/
function validateModelId(id: string) {
	if (!modelIdPattern.test(id)) {
		throw new Error(
			`Model ID must match pattern: ${modelIdPattern} (got "${id}")`,
		)
	}
}

interface DownloadTask {
	status: 'pending' | 'processing' | 'completed' | 'error'
	model: string
	handle: DownloadEngineNodejs
}

export interface LLMMStoreOptions {
	downloadConcurrency?: number
}

interface LLMMeta {
	gguf: Omit<GGUFMeta, 'tokenizer'>
	checksums?: Record<string, string>
}

export interface LLMStoreModelConfig extends LLMConfig {
	meta?: LLMMeta
}

export class LLMStore {
	downloadQueue: PQueue
	downloadTasks: DownloadTask[] = []
	modelsPath: string
	models: Record<string, LLMStoreModelConfig> = {}

	constructor(
		modelsPath: string,
		models: Record<string, LLMOptions>,
		options?: LLMMStoreOptions,
	) {
		this.downloadQueue = new PQueue({
			concurrency: options?.downloadConcurrency ?? 1,
		})
		this.modelsPath = modelsPath
		const storeModels: Record<string, LLMStoreModelConfig> = {}
		for (const modelId in models) {
			validateModelId(modelId)
			const modelOptions = models[modelId]
			if (!modelOptions.file && !modelOptions.url) {
				throw new Error(`Model ${modelId} must have either file or url`)
			}
			storeModels[modelId] = {
				id: modelId,
				minInstances: 0,
				maxInstances: 1,
				engineOptions: {},
				...modelOptions,
				file: this.resolveModelLocation({
					file: modelOptions.file,
					url: modelOptions.url,
				}),
			}
		}
		this.models = storeModels
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
				const fileExists = existsSync(model.file)
				const fileSize = fileExists ? statSync(model.file).size : 0
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
		}
	}

	private resolveModelLocation(options: { file?: string; url?: string }) {
		if (!options.file && !options.url) {
			throw new Error(`Must have either file or url`)
		}

		let autoSubPath = ''

		// make sure we create sub directories so models from different sources don't clash
		if (options.url) {
			const url = new URL(options.url)
			if (url.hostname === 'huggingface.co') {
				// TODO could consider accepting other url variants
				// Expecting URLs like
				// https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf
				const parts = url.pathname.split('/')
				if (parts.length < 6) {
					throw new Error(`Unexpected huggingface URL: ${options.url}`)
				}
				const org = parts[1]
				const repo = parts[2]
				const branch = parts[4]
				if (!org || !repo || !branch) {
					throw new Error(`Unexpected huggingface URL: ${options.url}`)
				}
				autoSubPath = 'huggingface/' + org + '/' + repo + '/' + branch
			} else {
				autoSubPath = url.hostname
			}
		}

		// resolve absolute path to file
		let absFilePath = ''
		if (options.file) {
			// if user explicitly provided a file path, use it
			if (path.isAbsolute(options.file)) {
				absFilePath = options.file
			} else {
				absFilePath = path.join(this.modelsPath, options.file)
			}
		} else if (options.url) {
			// otherwise create the default file location based on URL info
			const fileName = path.basename(new URL(options.url).pathname)
			absFilePath = path.join(this.modelsPath, autoSubPath, fileName)
		}

		return absFilePath
	}
}
