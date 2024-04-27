import PQueue from 'p-queue'
import { downloadFile } from 'ipull'
import type { ProgressStatusWithIndex } from 'ipull/dist/download/transfer-visualize/progress-statistics-builder.js'

interface ModelDownloaderOptions {
	concurrency?: number
}

interface DownloadOptions {
	file: string
	url: string
	onProgress?: (progress: ProgressStatusWithIndex) => void
}

export interface DownloadTask {
	file: string
	url: string
	progress: {
		loaded: number
		total: number
		percent: number
	}
}

export class ModelDownloader {
	queue: PQueue
	tasks: DownloadTask[] = []
	constructor(opts?: ModelDownloaderOptions) {
		this.queue = new PQueue({ concurrency: opts?.concurrency ?? 1 })
	}
	
	async enqueueDownload(opts: DownloadOptions, signal?: AbortSignal) {
		// if the file is already being downloaded, wait for it to complete
		const existingTask = this.tasks.find((t) => t.file === opts.file)
		if (existingTask) {
			return new Promise<void>((resolve, reject) => {
				const onComplete: any = (file: string) => {
					if (file === existingTask.file) {
						this.queue.off('completed', onComplete)
						resolve()
					}
				}
				this.queue.on('completed', onComplete)
			})
		}
		
		// otherwise, start a new download
		const task = {
			file: opts.file,
			url: opts.url,
			progress: {
				loaded: 0,
				total: 0,
				percent: 0,
			},
		}
		this.tasks.push(task)
		await this.queue.add(() => this.download({
			...opts,
			onProgress: (progress) => {
				task.progress.percent = progress.percentage
				task.progress.loaded = progress.transferredBytes
				task.progress.total = progress.totalBytes
				if (opts.onProgress) {
					opts.onProgress(progress)
				}
			},
		}, signal))
		this.tasks = this.tasks.filter((t) => t !== task)
		
	}
	
	private async download(opts: DownloadOptions, signal?: AbortSignal) {
		const downloader = await downloadFile({
			url: opts.url,
			savePath: opts.file,
			// parallelStreams: 3 // Number of parallel connections (default: 3)
		})
		if (signal) {
			signal.addEventListener('abort', () => {
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
			// if (opts.onError) {
			// 	opts.onError(error)
			// }
		}
		
		return opts.file
	}
}
