import { ModelOptions } from "#lllms/types/index.js"
import { builtInEngineList } from '#lllms/engines/index.js'

const modelIdPattern = /^[a-zA-Z0-9_\-\.]+$/
export function validateModelId(id: string) {
	if (!modelIdPattern.test(id)) {
		throw new Error(
			`Model "${id}" has invalid name; requires ${modelIdPattern}`,
		)
	}
}

export function validateModelOptions(id: string, modelOptions: ModelOptions) {
	validateModelId(id)
	if (!modelOptions.engine) {
		throw new Error(`Model "${id}" must have an engine`)
	}
	const isSourceMissing =
		!modelOptions.file && !modelOptions.url && !modelOptions.location
	if (builtInEngineList.includes(modelOptions.engine) && isSourceMissing) {
		throw new Error(`Model "${id}" must have either file or url`)
	}
	if (!modelOptions.task) {
		throw new Error(`Model "${id}" must have a task`)
	}
}