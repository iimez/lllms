import { promises as fs } from 'fs'
import { ggufMetadata } from 'hyllama'

export interface GGUFMeta {
	version: number
	general: {
		architecture: string
		name: string
		file_type: number
		quantization_version: number
	}
	llama?: unknown
	tokenizer?: {
		ggml?: {
			tokens?: string[]
			scores?: number[]
			token_type?: number[]
		}
	}
}

function structureGGUFMeta(metadata: Record<string, any>): GGUFMeta {
	const structuredMeta: any = {}

	for (const key in metadata) {
		const parts = key.split('.')
		let current: any = structuredMeta

		for (let i = 0; i < parts.length - 1; i++) {
			const part = parts[i]
			current[part] = current[part] || {}
			current = current[part]
		}

		current[parts[parts.length - 1]] = metadata[key]
	}

	return structuredMeta
}

export async function readGGUFMetaFromFile(file: string) {
	// Read first 10mb of gguf file
	const fd = await fs.open(file, 'r')
	const buffer = Buffer.alloc(10_000_000)
	await fd.read(buffer, 0, 10_000_000, 0)
	await fd.close()
	const { metadata, tensorInfos } = ggufMetadata(buffer.buffer)
	// console.log(metadata)
	// console.log(tensorInfos)
	return structureGGUFMeta(metadata)
}

export async function readGGUFMetaFromURL(url: string) {
	const headers = new Headers({ Range: 'bytes=0-10000000' })
	const res = await fetch(url, { headers })
	const arrayBuffer = await res.arrayBuffer()
	const { metadata, tensorInfos } = ggufMetadata(arrayBuffer)
	// console.log(metadata)
	// console.log(tensorInfos)

	return structureGGUFMeta(metadata)
}
