import path from 'node:path'
import fs from 'node:fs'
import { env } from '@huggingface/transformers'

// Currently unused Custom File Cache

/**
 * Mapping from file extensions to MIME types.
 */
const CONTENT_TYPE_MAP: Record<string, string> = {
	txt: 'text/plain',
	html: 'text/html',
	css: 'text/css',
	js: 'text/javascript',
	json: 'application/json',
	png: 'image/png',
	jpg: 'image/jpeg',
	jpeg: 'image/jpeg',
	gif: 'image/gif',
}

export class FileResponse {
	filePath: string | URL
	headers: Headers
	exists: boolean
	status: number
	statusText: string
	body: ReadableStream<Uint8Array> | null

	/**
	 * Creates a new `FileResponse` object.
	 * @param {string|URL} filePath
	 */
	constructor(filePath: string | URL) {
		this.filePath = filePath
		this.headers = new Headers()

		this.exists = fs.existsSync(filePath)
		if (this.exists) {
			this.status = 200
			this.statusText = 'OK'

			let stats = fs.statSync(filePath)
			this.headers.set('content-length', stats.size.toString())

			this.updateContentType()

			let self = this
			this.body = new ReadableStream({
				start(controller) {
					self.arrayBuffer().then((buffer) => {
						controller.enqueue(new Uint8Array(buffer))
						controller.close()
					})
				},
			})
		} else {
			this.status = 404
			this.statusText = 'Not Found'
			this.body = null
		}
	}

	/**
	 * Updates the 'content-type' header property of the response based on the extension of
	 * the file specified by the filePath property of the current object.
	 * @returns {void}
	 */
	updateContentType() {
		// Set content-type header based on file extension
		const extension = this.filePath.toString().split('.').pop()!.toLowerCase()
		this.headers.set(
			'content-type',
			CONTENT_TYPE_MAP[extension] ?? 'application/octet-stream',
		)
	}

	/**
	 * Clone the current FileResponse object.
	 * @returns {FileResponse} A new FileResponse object with the same properties as the current object.
	 */
	clone() {
		let response = new FileResponse(this.filePath)
		response.exists = this.exists
		response.status = this.status
		response.statusText = this.statusText
		response.headers = new Headers(this.headers)
		return response
	}

	/**
	 * Reads the contents of the file specified by the filePath property and returns a Promise that
	 * resolves with an ArrayBuffer containing the file's contents.
	 * @returns {Promise<ArrayBuffer>} A Promise that resolves with an ArrayBuffer containing the file's contents.
	 * @throws {Error} If the file cannot be read.
	 */
	async arrayBuffer() {
		const data = await fs.promises.readFile(this.filePath)
		return data.buffer
	}

	/**
	 * Reads the contents of the file specified by the filePath property and returns a Promise that
	 * resolves with a Blob containing the file's contents.
	 * @returns {Promise<Blob>} A Promise that resolves with a Blob containing the file's contents.
	 * @throws {Error} If the file cannot be read.
	 */
	async blob() {
		const data = await fs.promises.readFile(this.filePath)
		return new Blob([data], {
			type: this.headers.get('content-type') || undefined,
		})
	}

	/**
	 * Reads the contents of the file specified by the filePath property and returns a Promise that
	 * resolves with a string containing the file's contents.
	 * @returns {Promise<string>} A Promise that resolves with a string containing the file's contents.
	 * @throws {Error} If the file cannot be read.
	 */
	async text() {
		const data = await fs.promises.readFile(this.filePath, 'utf8')
		return data
	}

	/**
	 * Reads the contents of the file specified by the filePath property and returns a Promise that
	 * resolves with a parsed JavaScript object containing the file's contents.
	 *
	 * @returns {Promise<Object>} A Promise that resolves with a parsed JavaScript object containing the file's contents.
	 * @throws {Error} If the file cannot be read.
	 */
	async json() {
		return JSON.parse(await this.text())
	}
}

class TransformersFileCache {
	path: string
	prefix: string
	constructor(path: string) {
		this.path = path
		this.prefix = env.localModelPath
	}

	async match(request: string) {
		console.debug('FILE CACHE REQ', request)
		// throw new Error('FILE CACHE REQ')

		if (request.startsWith('http')) {
		} else {
			let filePath = request

			if (filePath.startsWith(this.prefix)) {
				filePath = filePath.replace(this.prefix, '')
				// next two segments are the model id
				const modelId = filePath.split('/').slice(0, 2).join('/')
				filePath = filePath.replace(modelId, '')
			}
			console.debug('FILE SUBPATH', filePath)
			filePath = path.join(this.path, filePath)

			console.debug('FILE ACTUAL', filePath)
			let file = new FileResponse(filePath)

			if (file.exists) {
				return file
			} else {
				return undefined
			}
		}
		return undefined
	}

	async put(request: string, response: FileResponse) {
		throw new Error('FILE CACHE put')
		const buffer = Buffer.from(await response.arrayBuffer())

		let outputPath = path.join(this.path, request)

		try {
			await fs.promises.mkdir(path.dirname(outputPath), { recursive: true })
			await fs.promises.writeFile(outputPath, buffer)
		} catch (err) {
			console.warn('An error occurred while writing the file to cache:', err)
		}
	}
}
