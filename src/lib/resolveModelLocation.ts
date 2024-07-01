import path from 'node:path'

export function resolveModelLocation(
	modelsPath: string,
	options: { file?: string; url?: string, location?: string, engine?: string },
) {
	
	if (options.location) {
		return options.location
	}
	
	// sub dir path so models from different sources don't clash
	let locationSubPath = ''

	if (options.url && !options.file) {
		const url = new URL(options.url)
		if (url.hostname === 'huggingface.co') {
			const parts = url.pathname.split('/')
			locationSubPath = 'huggingface/'
			const org = parts[1]
			const repo = parts[2]
			const branch = parts[4] || 'main'

			if (!org) {
				throw new Error(`Unexpected huggingface URL: ${options.url}`)
			}
			locationSubPath += org
			if (!repo) {
				throw new Error(`Unexpected huggingface URL: ${options.url}`)
			}
			locationSubPath += '/' + repo
			locationSubPath += '-' + branch
		} else {
			locationSubPath = url.hostname
		}
	}
	
	// TODO needs to be tested
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
		// if its a file, use it as is
		if (fileName.includes('.')) {
			absFilePath = path.join(modelsPath, locationSubPath, fileName)
		} else {
			// otherwise add a default file name
			absFilePath = path.join(modelsPath, locationSubPath)
		}
		// absFilePath = path.join(modelsPath, locationSubPath, fileName)
	}
	return absFilePath
}
