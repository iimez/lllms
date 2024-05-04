import path from 'node:path'
import { LLMConfig, LLMOptions } from '#lllms/types/index.js'

export function resolveModelConfig(models: Record<string, LLMOptions>, modelsDir: string) {
	const config: Record<string, LLMConfig> = {}
	for (const modelName in models) {
		validateModelName(modelName)
		const modelOpts = models[modelName]
		const file = resolveModelFile(modelOpts, modelsDir)
		const modelConfig = {
			minInstances: 0,
			maxInstances: 1,
			engineOptions: {},
			...modelOpts,
			file,
			name: modelName,
		}
		config[modelName] = modelConfig
	}
	return config
}

const modelNamePattern = /^[a-zA-Z0-9_:\-]+$/

function validateModelName(modelName: string) {
	if (!modelNamePattern.test(modelName)) {
		throw new Error(`Model name must match pattern: ${modelNamePattern} (got ${modelName})`)
	}
}

function resolveModelFile(opts: LLMOptions, modelsDir: string) {
	if (opts.file) {
		if (path.isAbsolute(opts.file)) {
			return opts.file
		} else {
			return path.join(modelsDir, opts.file)
		}
	}

	if (opts.url) {
		const url = new URL(opts.url)
		const modelName = path.basename(url.pathname)
		const modelPath = path.join(modelsDir, modelName)
		return modelPath
	}

	throw new Error('Model file or url is required')
}

