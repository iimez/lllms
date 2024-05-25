import type { AddressInfo } from 'node:net'
import { format as formatURL } from 'node:url'
import { serveLLMs, StandaloneServerOptions } from '#lllms/http.js'
import { LogLevels } from '#lllms/lib/logger.js'

// Currently only used for debugging. Do not use.

const serverOptions: StandaloneServerOptions = {
	listen: {
		port: 3000,
	},
	log: LogLevels.debug,
	// concurrency: 2,
	// maxDownloads: 2,
	models: {
		// 'phi3-mini-4k': {
		// 	url: 'https://gpt4all.io/models/gguf/Phi-3-mini-4k-instruct.Q4_0.gguf',
		// 	engine: 'gpt4all',
		// },
		// 'functionary': {
		// 	url: 'https://huggingface.co/meetkai/functionary-small-v2.4-GGUF/resolve/main/functionary-small-v2.4.Q4_0.gguf',
		// 	engineOptions: {
		// 		gpuLayers: 10,
		// 		gpu: true,
		// 	},
		// 	functions,
		// },
		// 'hermes': {
		// 	url: 'https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf',
		// 	minInstances: 1,
		// 	engineOptions: {
		// 		gpu: false,
		// 	}
		// },
		'nomic-text-embed': {
			url: 'https://gpt4all.io/models/gguf/nomic-embed-text-v1.f16.gguf',
			minInstances: 1,
			engine: 'gpt4all',
			task: 'embedding',
		},
		'llama3-8b': {
			url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			engine: 'node-llama-cpp',
			task: 'inference',
			// systemPrompt: 'Only answer in json format',
			// completionDefaults: {
			// 	grammar: 'json_object',
			// },
			// engineOptions: {
			// 	// memLock: true,
			// 	gpu: true,
			// },
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
