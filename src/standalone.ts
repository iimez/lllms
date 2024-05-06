import type { AddressInfo } from 'node:net'
import { format as formatURL } from 'node:url'
import { serveLLMs, StandaloneServerOptions } from '#lllms/http.js'
import { LogLevels } from '#lllms/lib/logger.js'

const serverOptions: StandaloneServerOptions = {
	listen: {
		port: 3000,
	},
	log: LogLevels.debug,
	// inferenceConcurrency: 2,
	// downloadConcurrency: 2,
	models: {
		// 'phi3-mini-4k': {
		// 	// url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
		// 	// engine: 'gpt4all',
		// 	url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
		// 	engine: 'node-llama-cpp',
		// 	// md5: 'cb68b653b24dc432b78b02df76921a54',
		// 	// sha256: '8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef',
		// 	minInstances: 1,
		// 	// maxInstances: 2,
		// 	// templateFormat: 'phi',
		// },
		'llama3-8b': {
			url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			engine: 'node-llama-cpp',
			// sha256: '19ded996fe6c60254dc7544d782276eff41046ed42aa5f2d0005dc457e5c0895',
			minInstances: 1,
			// engineOptions: {
			// 	// memLock: true,
			// 	gpu: true,
			// },
			// minInstances: 1,
			// url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			// engine: 'gpt4all',
		},
		// 'llama3-70b': {
		// 	url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q4_0.gguf',
		// 	engine: 'node-llama-cpp',
		// 	engineOptions: {
		// 		gpuLayers: 10,
		// 	},
		// 	// minInstances: 1,
		// },
		// 'orca-mini': {
		// 	file: 'orca-mini-3b-gguf2-q4_0.gguf',
		// 	engine: 'node-llama-cpp',
		// },
		// 'starcoder-7b': {
		// 	url: 'https://gpt4all.io/models/gguf/starcoder-newbpe-q4_0.gguf',
		// 	engine: 'gpt4all',
		// 	// minInstances: 1,
		// }
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
