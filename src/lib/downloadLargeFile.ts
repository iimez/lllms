import { downloadFile as createFileDownload } from 'ipull'
import { FileDownloadProgress } from '#lllms/types/index.js'

interface DownloadArgs {
	url: string
	file: string
	onProgress?: (progress: FileDownloadProgress) => void
	signal?: AbortSignal
}

export async function downloadLargeFile({
	url,
	file,
	onProgress,
	signal,
}: DownloadArgs) {
	const controller = await createFileDownload({
		url: url,
		savePath: file,
		
	})
	const progressInterval = setInterval(() => {
		if (onProgress) {
			onProgress({
				file: file,
				loadedBytes: controller.status.transferredBytes,
				totalBytes: controller.status.totalBytes,
			})
		}
	}, 1000)
	if (signal) {
		signal.addEventListener('abort', () => {
			controller.close()
		})
	}
	await controller.download()
	if (progressInterval) {
		clearInterval(progressInterval)
	}
}
