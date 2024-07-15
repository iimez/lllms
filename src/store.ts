import { promises as fs, existsSync } from 'node:fs'
import PQueue from 'p-queue'
import {
	FileDownloadProgress,
	ModelConfig,
	ModelEngine,
} from '#lllms/types/index.js'
import {
	Logger,
	LogLevels,
	LogLevel,
	createSublogger,
} from '#lllms/lib/logger.js'
import { mergeAbortSignals } from '#lllms/lib/util.js'

interface ModelFile {
	size: number
}

export interface StoredModel extends ModelConfig {
	meta?: unknown
	downloads?: Map<string, DownloadTracker>
	files?: Map<string, ModelFile>
	status: 'unloaded' | 'preparing' | 'ready' | 'error'
}

export interface ModelStoreOptions {
	modelsPath: string
	models: Record<string, ModelConfig>
	prepareConcurrency?: number
	log?: Logger | LogLevel
}

export class ModelStore {
	prepareQueue: PQueue
	models: Record<string, StoredModel> = {}
	engines?: Record<string, ModelEngine>
	private prepareController: AbortController
	private modelsPath: string
	private log: Logger

	constructor(options: ModelStoreOptions) {
		this.prepareController = new AbortController()
		this.log = createSublogger(options.log)
		this.prepareQueue = new PQueue({
			concurrency: options.prepareConcurrency ?? 2,
		})
		this.modelsPath = options.modelsPath
		this.models = Object.fromEntries(
			Object.entries(options.models).map(([modelId, model]) => [
				modelId,
				{
					...model,
					status: 'unloaded',
				},
			]),
		)
	}

	async init(engines: Record<string, ModelEngine>) {
		this.engines = engines
		if (!existsSync(this.modelsPath)) {
			await fs.mkdir(this.modelsPath, { recursive: true })
		}

		const blockingPromises = []
		for (const modelId in this.models) {
			const model = this.models[modelId]
			if (model.prepare === 'blocking' || model.minInstances > 0) {
				blockingPromises.push(this.prepareModel(modelId))
			} else if (model.prepare === 'async') {
				this.prepareModel(modelId)
			}
		}
		await Promise.all(blockingPromises)
	}

	dispose() {
		this.prepareController.abort()
	}

	private onDownloadProgress(
		modelId: string,
		progress: { file: string; loadedBytes: number; totalBytes: number },
	) {
		const model = this.models[modelId]
		if (!model.downloads) {
			model.downloads = new Map()
		}
		if (progress.totalBytes && progress.totalBytes === progress.loadedBytes) {
			model.downloads.delete(progress.file)
		} else if (model.downloads.has(progress.file)) {
			const tracker = model.downloads.get(progress.file)!
			tracker.pushProgress(progress)
		} else {
			const tracker = new DownloadTracker(5000)
			tracker.pushProgress(progress)
			model.downloads.set(progress.file, tracker)
		}
	}

	// makes sure all required files for the model exist and are valid
	// checking model checksums and reading metadata is model + engine specific and can be slow
	async prepareModel(modelId: string, signal?: AbortSignal) {
		const model = this.models[modelId]
		if (!this.engines) {
			throw new Error('No engines available - did you call init()?')
		}
		model.status = 'preparing'
		const engine = this.engines[model.engine]
		this.log(LogLevels.info, 'Preparing model', {
			model: modelId,
			task: model.task,
		})

		await this.prepareQueue.add(async () => {
			if (!('prepareModel' in engine)) {
				model.status = 'ready'
				return model
			}
			const logProgressInterval = setInterval(() => {
				const progress = Array.from(model.downloads?.values() ?? [])
					.map((tracker) => tracker.getStatus())
					.reduce(
						(acc, status) => {
							acc.loadedBytes += status?.loadedBytes || 0
							acc.totalBytes += status?.totalBytes || 0
							return acc
						},
						{ loadedBytes: 0, totalBytes: 0 },
					)
				if (progress.totalBytes) {
					const percent = (progress.loadedBytes / progress.totalBytes) * 100
					this.log(LogLevels.info, `downloading ${percent.toFixed(1)}%`, {
						model: modelId,
					})
				}
			}, 10000)
			try {
				const modelMeta = await engine.prepareModel(
					{ config: model, log: this.log },
					(progress) => {
						this.onDownloadProgress(model.id, progress)
					},
					mergeAbortSignals([signal, this.prepareController.signal]),
				)
				model.meta = modelMeta
				model.status = 'ready'
				this.log(LogLevels.info, 'Model ready', {
					model: modelId,
					task: model.task,
				})
			} catch (error) {
				this.log(LogLevels.error, 'Error preparing model', {
					model: modelId,
					error: error,
				})
				model.status = 'error'
			} finally {
				clearInterval(logProgressInterval)
			}
			return model
		})
	}

	getStatus() {
		const formatFloat = (num?: number) => parseFloat(num?.toFixed(2) || '0')
		const storeStatusInfo = Object.fromEntries(
			Object.entries(this.models).map(([modelId, model]) => {
				let downloads: any = undefined
				if (model.downloads) {
					downloads = [...model.downloads].reduce<any>(
						(acc, [key, download]) => {
							const status = download.getStatus()
							const latestState =
								download.progressBuffer[download.progressBuffer.length - 1]
							// console.log('latestState', latestState)
							acc.push({
								file: key,
								...status,
								percent: formatFloat(status?.percent),
								speed: formatFloat(status?.speed),
								eta: formatFloat(status?.eta),
								// latest: latestState,
							})
							return acc
						},
						[],
					)
				}
				return [
					modelId,
					{
						engine: model.engine,
						device: model.device,
						minInstances: model.minInstances,
						maxInstances: model.maxInstances,
						status: model.status,
						downloads,
					},
				]
			}),
		)
		return storeStatusInfo
	}
}

type ProgressState = {
	loadedBytes: number
	totalBytes: number
	timestamp: number // in milliseconds
}

type DownloadStatus = {
	percent: number
	speed: number
	eta: number
	loadedBytes: number
	totalBytes: number
}

class DownloadTracker {
	progressBuffer: ProgressState[] = []
	private timeWindow: number

	constructor(timeWindow: number = 1000) {
		this.timeWindow = timeWindow
	}

	pushProgress({ loadedBytes, totalBytes }: FileDownloadProgress): void {
		const timestamp = Date.now()
		this.progressBuffer.push({ loadedBytes, totalBytes, timestamp })
		this.cleanup()
	}

	private cleanup(): void {
		const cutoffTime = Date.now() - this.timeWindow
		this.progressBuffer = this.progressBuffer.filter(
			(item) => item.timestamp >= cutoffTime,
		)
	}

	getStatus(): DownloadStatus | null {
		if (this.progressBuffer.length < 2) {
			return null // Not enough data to calculate speed and ETA
		}

		const latestState = this.progressBuffer[this.progressBuffer.length - 1]
		const previousState = this.progressBuffer[0] // oldest state within the time window

		const bytesLoaded = latestState.loadedBytes - previousState.loadedBytes
		const timeElapsed = latestState.timestamp - previousState.timestamp // in milliseconds

		const speed = bytesLoaded / (timeElapsed / 1000) // bytes per second
		const remainingBytes = latestState.totalBytes - latestState.loadedBytes
		const eta = remainingBytes / speed // in seconds

		return {
			speed,
			eta,
			percent: latestState.loadedBytes / latestState.totalBytes,
			loadedBytes: latestState.loadedBytes,
			totalBytes: latestState.totalBytes,
		}
	}
}
