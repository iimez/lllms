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
	models: {
		'nomic-text-embed': {
			url: 'https://gpt4all.io/models/gguf/nomic-embed-text-v1.f16.gguf',
			minInstances: 1,
			engine: 'gpt4all',
			task: 'embedding',
		},
		'phi-3-mini-128k': {
			url: 'https://huggingface.co/QuantFactory/Phi-3-mini-128k-instruct-GGUF/resolve/main/Phi-3-mini-128k-instruct.Q4_0.gguf',
			engine: 'node-llama-cpp',
			task: 'text-completion',
			prepare: 'blocking',
			preload: {
				// documentFunctions: true,
				messages: [
					{
						role: 'user',
						content: 'What is the meaning of life?',
					},
					{
						role: 'assistant',
						content: 'The meaning of life is 41.',
					},
				],
			}
		},
		'llama3-8b': {
			url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			sha256: '1977ae6185ef5bc476e27db85bb3d79ca4bd87e7b03399083c297d9c612d334c',
			// url: 'https://huggingface.co/mradermacher/Llama-3-Smaug-8B-OAS-i1-GGUF/blob/main/Llama-3-Smaug-8B-OAS.i1-Q4_K_M.gguf',
			engine: 'node-llama-cpp',
			task: 'text-completion',
			prepare: 'async',
			engineOptions: {
				gpu: 'metal',
			}
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
			// https://huggingface.co/bartowski/Smaug-Llama-3-70B-Instruct-GGUF/tree/main/Smaug-Llama-3-70B-Instruct-Q8_0.gguf
		// 	url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q4_0.gguf',
		// 	engine: 'node-llama-cpp',
		// 	engineOptions: {
		// 		gpuLayers: 10,
		// 	},
		// 	// minInstances: 1,
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
