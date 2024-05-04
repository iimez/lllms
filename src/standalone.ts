import type { AddressInfo } from 'node:net'
import { format as formatURL } from 'node:url'
import { serveLLMs, StandaloneServerOptions } from '#lllms/server.js'
import { LogLevels } from '#lllms/lib/logger.js'

const serverOptions: StandaloneServerOptions = {
	listen: {
		port: 3000,
	},
	logLevel: LogLevels.debug,
	concurrency: 2,
	models: {
		'phi3-mini-4k': {
			url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
			engine: 'gpt4all',
			// url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
			// engine: 'node-llama-cpp',
			// minInstances: 1,
			// maxInstances: 2,
			// templateFormat: 'phi',
		},
		'llama3-8b': {
			url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			minInstances: 1,
			
			engine: 'node-llama-cpp',
			engineOptions: {
				memLock: true,
				gpu: true,
				// gpuLayers: 4,
				batchSize: 512,
			},
			
			// url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			// engine: 'gpt4all',
		},
		'llama3-70b': {
			url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q4_0.gguf',
			engine: 'node-llama-cpp',
		},
		'starcoder-7b': {
			url: 'https://gpt4all.io/models/gguf/starcoder-newbpe-q4_0.gguf',
			engine: 'gpt4all',
			// minInstances: 1,
		}
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
