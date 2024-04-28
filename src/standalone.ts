import type { AddressInfo } from 'node:net'
import { format as formatURL } from 'node:url'
import { serveLLMs, StandaloneServerOptions } from './server.js'

const serverOptions: StandaloneServerOptions = {
	listen: {
		port: 3000,
	},
	concurrency: 2,
	models: {
		'phi3-mini': {
			url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
			engine: 'gpt4all',
			// url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
			// engine: 'node-llama-cpp',
			minInstances: 1,
			maxInstances: 2,
			templateFormat: 'phi',
		},
		'orca-3b': {
			url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
			maxInstances: 2,
			engine: 'gpt4all',
			templateFormat: 'alpaca',
		},
		'llama3-8b': {
			url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			engine: 'node-llama-cpp',
			// url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			// engine: 'gpt4all',
		},
		'llama3-70b': {
			url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q4_0.gguf',
			engine: 'node-llama-cpp',
		},
	},
}

async function main() {
	const server = await serveLLMs(serverOptions)
	const { address, port } = server.address() as AddressInfo
	const hostname = address === '' || address === '::' ? 'localhost' : address
	const url = formatURL({
		protocol: 'http',
		hostname,
		port,
		pathname: '/',
	})
	console.log(`Server listening at ${url}`)
}

main().catch((err: Error) => {
	console.error(err)
	process.exit(1)
})

process.on('unhandledRejection', (err) => {
	console.error('Unhandled rejection:', err)
	process.exit(1)
})
