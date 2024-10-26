import { downloadFile as createFileDownload } from 'ipull'
import fs from 'node:fs'
import { FileDownloadProgress } from '#package/types/index.js'

interface DownloadArgs {
	url: string
	file: string
	onProgress?: (progress: FileDownloadProgress) => void
	signal?: AbortSignal
}

export async function downloadModelFile({
	url,
	file,
	onProgress,
	signal,
}: DownloadArgs) {
	let downloadUrl = url
	const parsedUrl = new URL(url)
	if (parsedUrl.hostname === 'huggingface.co') {
		const pathnameParts = parsedUrl.pathname.split('/')
		if (pathnameParts.length > 3 && pathnameParts[3] === 'blob') {
			const newUrl = new URL(url)
			pathnameParts[3] = 'resolve'
			newUrl.pathname = pathnameParts.join('/')
			if (newUrl.searchParams.get('download') !== 'true') {
				newUrl.searchParams.set('download', 'true')
			}
			downloadUrl = newUrl.href
		}
	}
	const controller = await createFileDownload({
		url: downloadUrl,
		savePath: file,
		skipExisting: true,
	})
	
	let partialBytes = 0
	if (fs.existsSync(file)) {
		partialBytes = fs.statSync(file).size
	}
	const progressInterval = setInterval(() => {
		if (onProgress) {
			onProgress({
				file: file,
				loadedBytes: controller.status.transferredBytes + partialBytes,
				totalBytes: controller.status.totalBytes,
			})
		}
	}, 3000)
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
