#!/usr/bin/env node
import path from 'node:path'
import os from 'node:os'
import fs from 'node:fs'
import { downloadFile, downloadSequence } from 'ipull'

const destDir = path.resolve(os.homedir(), '.cache/lllms')
const testModels = [
	{
		url: 'https://huggingface.co/mradermacher/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf?download=true',
		dest: path.resolve(
			destDir,
			'huggingface/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf',
		),
	},
	{
		url: 'https://huggingface.co/meetkai/functionary-small-v2.5-GGUF/resolve/main/functionary-small-v2.5.Q4_0.gguf?download=true',
		dest: path.resolve(
			destDir,
			'huggingface/meetkai/functionary-small-v2.5-GGUF-main/functionary-small-v2.5.Q4_0.gguf',
		),
	},
	{
		url: 'https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf?download=true',
		dest: path.resolve(
			destDir,
			'huggingface/bartowski/Phi-3.1-mini-4k-instruct-GGUF-main/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf',
		),
	},
	{
		url: 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf?download=true',
		dest: path.resolve(
			destDir,
			'huggingface/nomic-ai/nomic-embed-text-v1.5-GGUF-main/nomic-embed-text-v1.5.Q8_0.gguf',
		),
	},
	{
		url: 'https://gpt4all.io/models/gguf/nomic-embed-text-v1.f16.gguf',
		dest: path.resolve(destDir, 'gpt4all.io/nomic-embed-text-v1.f16.gguf'),
	},
	{
		url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
		dest: path.resolve(destDir, 'gpt4all.io/Phi-3-mini-4k-instruct.Q4_0.gguf'),
	},
	{
		url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
		dest: path.resolve(destDir, 'gpt4all.io/Meta-Llama-3-8B-Instruct.Q4_0.gguf'),
	},
]

const pendingDownloads = []

for (const model of testModels) {
	if (!fs.existsSync(model.dest)) {
		pendingDownloads.push(
			downloadFile({
				url: model.url,
				directory: path.dirname(model.dest),
				fileName: path.basename(model.dest),
			}),
		)
	}
}

if (pendingDownloads.length === 0) {
	console.info('All models already downloaded')
	process.exit(0)
}

console.info(
	`Downloading ${pendingDownloads.length} model${
		pendingDownloads.length === 1 ? '' : 's'
	}`,
)
const downloader = await downloadSequence(
	{
		cliProgress: true,
		parallelDownloads: 4,
	},
	...pendingDownloads,
)
await downloader.download()
