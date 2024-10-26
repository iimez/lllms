import * as StableDiffusion from '@lmagder/node-stable-diffusion-cpp'
import { gguf } from '@huggingface/gguf'
import sharp from 'sharp'
import fs from 'node:fs'
import path from 'node:path'
import {
	EngineContext,
	FileDownloadProgress,
	ModelConfig,
	EngineTextToImageResult,
	ModelFileSource,
	EngineTextToImageArgs,
	Image,
	EngineImageToImageArgs,
} from '#package/types/index.js'
import { LogLevel, LogLevels } from '#package/lib/logger.js'
import { downloadModelFile } from '#package/lib/downloadModelFile.js'
import { acquireFileLock } from '#package/lib/acquireFileLock.js'
import { calculateFileChecksum } from '#package/lib/calculateFileChecksum.js'
import { getRandomNumber } from '#package/lib/util.js'
import {
	StableDiffusionSamplingMethod,
	StableDiffusionSchedule,
	StableDiffusionWeightType,
} from './types.js'

export interface StableDiffusionInstance {
	context: StableDiffusion.Context
}

export interface StableDiffusionModelConfig extends ModelConfig {
	location: string
	sha256?: string
	clipL?: ModelFileSource
	clipG?: ModelFileSource
	vae?: ModelFileSource
	t5xxl?: ModelFileSource
	controlNet?: ModelFileSource
	taesd?: ModelFileSource
	diffusionModel?: boolean
	model?: ModelFileSource
	loras?: ModelFileSource[]
	samplingMethod?: StableDiffusionSamplingMethod
	weightType?: StableDiffusionWeightType
	schedule?: StableDiffusionSchedule
	device?: {
		gpu?: boolean | 'auto' | (string & {})
		cpuThreads?: number
	}
}

interface StableDiffusionModelMeta {
	gguf: any
}

export const autoGpu = true

function getModelFileName(opts: ModelFileSource) {
	if (opts.url) {
		const parsedUrl = new URL(opts.url)
		return path.basename(parsedUrl.pathname)
	}
	if (opts.file) {
		return path.basename(opts.file)
	}
	throw new Error('Invalid file options')
}

async function validateModelFiles(config: StableDiffusionModelConfig) {
	if (!fs.existsSync(config.location)) {
		return 'model file missing'
	}

	const ipullFile = config.location + '.ipull'
	if (fs.existsSync(ipullFile)) {
		return 'pending download'
	}

	const modelDir = path.dirname(config.location)
	const validateFile = async (name: string, src: ModelFileSource) => {
		const file = path.join(modelDir, getModelFileName(src))
		if (!fs.existsSync(file)) {
			return `${name} file missing at ${file}`
		}
		if (src.sha256) {
			const fileHash = await calculateFileChecksum(file, 'sha256')
			if (fileHash !== src.sha256) {
				return `file sha256 checksum mismatch: expected ${src.sha256} got ${fileHash} in ${file}`
			}
		}
		return undefined
	}
	if (config.clipL) {
		const clipLError = await validateFile('clipL', config.clipL)
		if (clipLError) {
			return clipLError
		}
	}
	if (config.clipG) {
		const clipGError = await validateFile('clipG', config.clipG)
		if (clipGError) {
			return clipGError
		}
	}
	if (config.vae) {
		const vaeError = await validateFile('vae', config.vae)
		if (vaeError) {
			return vaeError
		}
	}
	if (config.t5xxl) {
		const t5xxlError = await validateFile('t5xxl', config.t5xxl)
		if (t5xxlError) {
			return t5xxlError
		}
	}
	if (config.controlNet) {
		const controlNetError = await validateFile('controlNet', config.controlNet)
		if (controlNetError) {
			return controlNetError
		}
	}
	if (config.taesd) {
		const taesdError = await validateFile('taesd', config.taesd)
		if (taesdError) {
			return taesdError
		}
	}

	if (config.sha256) {
		const fileHash = await calculateFileChecksum(config.location, 'sha256')
		if (fileHash !== config.sha256) {
			return `model file sha256 checksum mismatch: expected ${config.sha256} got ${fileHash}`
		}
	}

	const loraDir = path.join(path.dirname(config.location), 'lora')
	if (config.loras) {
		for (const lora of config.loras) {
			const loraFile = path.join(loraDir, getModelFileName(lora))
			if (!fs.existsSync(loraFile)) {
				return `lora file missing: ${loraFile}`
			}
		}
	}

	return undefined
}

export async function prepareModel(
	{
		config,
		log,
	}: EngineContext<StableDiffusionModelConfig, StableDiffusionModelMeta>,
	onProgress?: (progress: FileDownloadProgress) => void,
	signal?: AbortSignal,
) {
	fs.mkdirSync(path.dirname(config.location), { recursive: true })
	const clearFileLock = await acquireFileLock(config.location, signal)
	if (signal?.aborted) {
		return
	}
	log(
		LogLevels.info,
		`Preparing stable-diffusion model at ${config.location}`,
		{
			model: config.id,
		},
	)

	const downloadModel = (url: string, reason: string) => {
		log(LogLevels.info, `${reason} - Downloading model files`, {
			model: config.id,
			url: config.url,
			location: config.location,
		})
		const downloadPromises = []
		const modelDir = path.dirname(config.location)
		downloadPromises.push(
			downloadModelFile({
				url: url,
				file: config.location,
				onProgress,
				signal,
			}),
		)
		const filesToDownload = [
			config.clipL,
			config.clipG,
			config.vae,
			config.t5xxl,
			config.controlNet,
			config.taesd,
		]
		for (const file of filesToDownload) {
			if (!file?.url) {
				continue
			}
			downloadPromises.push(
				downloadModelFile({
					url: file.url,
					file: path.join(modelDir, getModelFileName(file)),
					onProgress,
					signal,
				}),
			)
		}
		if (config.loras) {
			const loraDir = path.join(path.dirname(config.location), 'loras')
			fs.mkdirSync(loraDir, { recursive: true })
			for (const lora of config.loras) {
				if (!lora.url) {
					continue
				}
				downloadPromises.push(
					downloadModelFile({
						url: lora.url,
						file: path.join(loraDir, getModelFileName(lora)),
						onProgress,
						signal,
					}),
				)
			}
		}

		return Promise.all(downloadPromises)
	}

	try {
		const validationError = await validateModelFiles(config)
		if (signal?.aborted) {
			return
		}
		if (validationError) {
			if (config.url) {
				await downloadModel(config.url, validationError)
			} else {
				throw new Error(`Model files are invalid: ${validationError}`)
			}
		}

		const finalValidationError = await validateModelFiles(config)
		if (finalValidationError) {
			throw new Error(`Model files are invalid: ${finalValidationError}`)
		}
		clearFileLock()

		const result: any = {}

		if (config.location.endsWith('.gguf')) {
			const { metadata, tensorInfos } = await gguf(config.location, {
				allowLocalFile: true,
			})
			result.gguf = metadata
		}
		return result
	} catch (err) {
		clearFileLock()
		throw err
	}
}

function parseQuantization(filename: string): string | null {
	// Regular expressions to match different quantization patterns
	const regexPatterns = [
		/q(\d+)_(\d+)/i, // q4_0
		/[-_\.](f16|f32|int8|int4)/i, // f16, f32, int8, int4
		/[-_\.](fp16|fp32)/i, // fp16, fp32
	]

	for (const regex of regexPatterns) {
		const match = filename.match(regex)
		console.debug({
			regex: regex.toString(),
			match,
		})
		if (match) {
			// If there's a match, return the full matched quantization string
			// Remove leading dash if present, convert to uppercase
			return match[0].replace(/^[-_]/, '').replace(/fp/i, 'f').toLowerCase()
		}
	}
	return null
}

function getWeightType(key: string): number | undefined {
	const weightKey = key.toUpperCase() as keyof typeof StableDiffusion.Type
	if (weightKey in StableDiffusion.Type) {
		return StableDiffusion.Type[weightKey]
	}
	console.warn('Unknown weight type', weightKey)
	return undefined
}

function getSamplingMethod(
	method?: string,
): StableDiffusion.SampleMethod | undefined {
	switch (method) {
		case 'euler':
			return StableDiffusion.SampleMethod.Euler
		case 'euler_a':
			return StableDiffusion.SampleMethod.EulerA
		case 'lcm':
			return StableDiffusion.SampleMethod.LCM
		case 'heun':
			return StableDiffusion.SampleMethod.Heun
		case 'dpm2':
			return StableDiffusion.SampleMethod.DPM2
		case 'dpm++2s_a':
			return StableDiffusion.SampleMethod.DPMPP2SA
		case 'dpm++2m':
			return StableDiffusion.SampleMethod.DPMPP2M
		case 'dpm++2mv2':
			return StableDiffusion.SampleMethod.DPMPP2Mv2
		case 'ipndm':
			// @ts-ignore
			return StableDiffusion.SampleMethod.IPNDM
		case 'ipndm_v':
			// @ts-ignore
			return StableDiffusion.SampleMethod.IPNDMV
	}
	console.warn('Unknown sampling method', method)
	return undefined
}

export async function createInstance(
	{ config, log }: EngineContext<StableDiffusionModelConfig>,
	signal?: AbortSignal,
) {
	log(LogLevels.debug, 'Load Stable Diffusion model', config)

	const handleLog = (level: string, message: string) => {
		log(level as LogLevel, message)
	}
	const handleProgress = (step: number, steps: number, time: number) => {
		log(LogLevels.debug, `Progress: ${step}/${steps} (${time}ms)`)
	}
	const modelDir = path.dirname(config.location)

	// stable-diffusion.cpp example https://github.com/leejet/stable-diffusion.cpp/blob/14206fd48832ab600d9db75f15acb5062ae2c296/examples/cli/main.cpp#L766
	// gpu/device https://github.com/search?q=repo%3Aleejet%2Fstable-diffusion.cpp+CUDA_VISIBLE_DEVICES&type=issues
	// createContext https://github.com/lmagder/node-stable-diffusion.cpp/blob/b4c7bd7786320d2e1ce13763c31bc9e32c061a3e/src/NodeModule.cpp#L335

	const vaeFilePath = config.vae
		? path.join(modelDir, getModelFileName(config.vae))
		: undefined
	const clipLFilePath = config.clipL
		? path.join(modelDir, getModelFileName(config.clipL))
		: undefined
	const clipGFilePath = config.clipG
		? path.join(modelDir, getModelFileName(config.clipG))
		: undefined
	const t5xxlFilePath = config.t5xxl
		? path.join(modelDir, getModelFileName(config.t5xxl))
		: undefined
	const controlNetFilePath = config.controlNet
		? path.join(modelDir, getModelFileName(config.controlNet))
		: undefined
	const taesdFilePath = config.taesd
		? path.join(modelDir, getModelFileName(config.taesd))
		: undefined

	let weightType = config.weightType
		? getWeightType(config.weightType)
		: undefined
	if (typeof weightType === 'undefined') {
		const quantization = parseQuantization(config.location)
		if (quantization) {
			weightType = getWeightType(quantization)
		}
	}

	if (typeof weightType === 'undefined') {
		throw new Error('Failed to determine weight type / quantization')
	}

	const loraDir = path.join(path.dirname(config.location), 'loras')
	const contextParams = {
		model: !config.diffusionModel ? config.location : undefined,
		diffusionModel: config.diffusionModel ? config.location : undefined,
		numThreads: config.device?.cpuThreads,
		vae: vaeFilePath,
		clipL: clipLFilePath,
		clipG: clipGFilePath,
		t5xxl: t5xxlFilePath,
		controlNet: controlNetFilePath,
		taesd: taesdFilePath,
		weightType: weightType,
		loraDir: loraDir,
	}
	log(LogLevels.debug, 'Creating context with', contextParams)
	const context = await StableDiffusion.createContext(
		// @ts-ignore
		contextParams,
		handleLog,
		handleProgress,
	)

	return {
		context,
	}
}

export async function processTextToImageTask(
	{ request, config, log }: EngineTextToImageArgs<StableDiffusionModelConfig>,
	instance: StableDiffusionInstance,
	signal?: AbortSignal,
): Promise<EngineTextToImageResult> {
	const seed = request.seed ?? getRandomNumber(0, 1000000)
	const results = await instance.context.txt2img({
		prompt: request.prompt,
		negativePrompt: request.negativePrompt,
		width: request.width || 512,
		height: request.height || 512,
		batchCount: request.batchCount,
		sampleMethod: getSamplingMethod(
			request.samplingMethod || config.samplingMethod,
		),
		sampleSteps: request.sampleSteps,
		cfgScale: request.cfgScale,
		// @ts-ignore
		guidance: request.guidance,
		styleRatio: request.styleRatio,
		controlStrength: request.controlStrength,
		normalizeInput: false,
		seed,
	})

	const images: Image[] = []
	for (const [idx, img] of results.entries()) {
		images.push({
			handle: sharp(img.data, {
				raw: {
					width: img.width,
					height: img.height,
					channels: img.channel,
				},
			}),
			width: img.width,
			height: img.height,
			channels: img.channel,
		})
	}
	if (!images.length) {
		throw new Error('No images generated')
	}
	return {
		images: images,
		seed,
	}
}

export async function processImageToImageTask(
	{ request, config, log }: EngineImageToImageArgs<StableDiffusionModelConfig>,
	instance: StableDiffusionInstance,
	signal?: AbortSignal,
): Promise<EngineTextToImageResult> {
	const seed = request.seed ?? getRandomNumber(0, 1000000)
	console.debug('processImageToImageTask', {
		width: request.image.width,
		height: request.image.height,
		channel: request.image.channels as 3 | 4,
	})
	const initImage = {
		data: await request.image.handle.raw().toBuffer(),
		width: request.image.width,
		height: request.image.height,
		channel: request.image.channels as 3 | 4,
	}
	const results = await instance.context.img2img({
		initImage,
		prompt: request.prompt,
		width: request.width || 512,
		height: request.height || 512,
		batchCount: request.batchCount,
		sampleMethod: getSamplingMethod(
			request.samplingMethod || config.samplingMethod,
		),
		cfgScale: request.cfgScale,
		sampleSteps: request.sampleSteps,
		// @ts-ignore
		guidance: request.guidance,
		strength: request.strength,
		styleRatio: request.styleRatio,
		controlStrength: request.controlStrength,
		seed,
	})

	const images: Image[] = []
	// to sharp
	// const imagePromises = results.map(async (img, idx) => {
	// 	return await sharp(img.data, {
	// 			raw: {
	// 				width: img.width,
	// 				height: img.height,
	// 				channels: img.channel,
	// 			},
	// 		})
	// 	})

	for (const [idx, img] of results.entries()) {
		console.debug('img', {
			id: idx,
			width: img.width,
			height: img.height,
			channels: img.channel,
		})

		images.push({
			handle: sharp(img.data, {
				raw: {
					width: img.width,
					height: img.height,
					channels: img.channel,
				},
			}),
			width: img.width,
			height: img.height,
			channels: img.channel,
		})

		// images.push({
		// 	data: img.data,
		// 	width: img.width,
		// 	height: img.height,
		// 	channels: img.channel,
		// })
	}
	if (!images.length) {
		throw new Error('No images generated')
	}
	return {
		images: images,
		seed,
	}
}
