import fs from 'fs'
import { TransformersJsModelConfig } from "./engine.js"


export function checkModelExists(config: TransformersJsModelConfig) {
	if (!fs.existsSync(config.location)) {
		return false
	}
	if (!fs.existsSync(config.location + '/onnx')) {
		return false
	}

	const checkDTypeExists = (dtype: string | Record<string, string>) => {
		// TODO needs more work, does not handle all cases
		if (typeof dtype === 'string') {
			if (dtype === 'fp32') {
				const expectedFile = `${config.location}/onnx/encoder_model.onnx`
				if (!fs.existsSync(expectedFile)) {
					console.debug('missing', dtype, expectedFile)
					return false
				}
			}
		} else if (typeof dtype === 'object') {
			for (const fileName in dtype) {
				const dataType = dtype[fileName]
				let expectedFile = `${config.location}/onnx/${fileName}_${dataType}.onnx`
				if (dataType === 'fp32') {
					expectedFile = `${config.location}/onnx/${fileName}.onnx`
				}
				if (!fs.existsSync(expectedFile)) {
					console.debug('missing', dtype, expectedFile)
					return false
				}
			}
		}
		return true
	}
	if (config.textModel?.dtype) {
		const notExisting = checkDTypeExists(config.textModel.dtype)
		if (!notExisting) {
			return false
		}
	}
	if (config.visionModel?.dtype) {
		const notExisting = checkDTypeExists(config.visionModel.dtype)
		if (!notExisting) {
			return false
		}
	}
	if (config.speechModel?.dtype) {
		const notExisting = checkDTypeExists(config.speechModel.dtype)
		if (!notExisting) {
			return false
		}
	}
	return true
}

export async function remoteFileExists(url: string): Promise<boolean> {
	try {
		const response = await fetch(url, { method: 'HEAD' })
		return response.ok
	} catch (error) {
		console.error('Error checking remote file:', error)
		return false
	}
}

export function parseModelUrl(url: string) {
	const parsedUrl = new URL(url)
	const urlSegments = parsedUrl.pathname.split('/')
	const org = urlSegments[1]
	const repo = urlSegments[2]
	const branch = urlSegments[4] || 'main'
	return {
		modelId: `${org}/${repo}`,
		branch,
	}
}