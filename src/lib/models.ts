import path from 'node:path'

const modelIdPattern = /^[a-zA-Z0-9_:\-]+$/
export function validateModelId(id: string) {
	if (!modelIdPattern.test(id)) {
		throw new Error(
			`Model ID must match pattern: ${modelIdPattern} (got "${id}")`,
		)
	}
}

// see node-llama-cpp src/gguf/utils/normalizeGgufDownloadUrl.ts
export function resolveModelUrl(url: string) {
	const parsedUrl = new URL(url)
	if (parsedUrl.hostname === 'huggingface.co') {
		const pathnameParts = parsedUrl.pathname.split('/')
		if (pathnameParts.length > 3) {
			const newUrl = new URL(url)
			if (pathnameParts[3] === 'blob' || pathnameParts[3] === 'raw') {
				pathnameParts[3] = 'resolve'
			}
			newUrl.pathname = pathnameParts.join('/')
			if (newUrl.searchParams.get('download') !== 'true') {
				newUrl.searchParams.set('download', 'true')
			}
			return newUrl.href
		}
	}
	return url
}

export function resolveModelFile(
	modelsPath: string,
	options: { file?: string; url?: string },
) {
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
			if (!org || !repo || !branch || parts[3] !== 'resolve') {
				throw new Error(`Unexpected huggingface URL: ${options.url}`)
			}
			autoSubPath = 'huggingface/' + org + '/' + repo + '-' + branch
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
			absFilePath = path.join(modelsPath, options.file)
		}
	} else if (options.url) {
		// otherwise create the default file location based on URL info
		const fileName = path.basename(new URL(options.url).pathname)
		absFilePath = path.join(modelsPath, autoSubPath, fileName)
	}

	return absFilePath
}
