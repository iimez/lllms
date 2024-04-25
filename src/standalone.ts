import { createLLMServer, LLMServerOptions } from './server.js'

const config: LLMServerOptions = {
	concurrency: 1,
	models: {
		'orca-3b': {
			url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
			minInstances: 1,
			engine: 'gpt4all',
			templateFormat: 'alpaca',
		},
		phi3: {
			url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
			minInstances: 1,
			engine: 'node-llama-cpp',
		},
		'llama3-8b': {
			url: 'https://huggingface.co/NousResearch/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B-Q4_K_M.gguf',
			engine: 'node-llama-cpp',
			templateFormat: 'llama3',
		},
		// 'llama3-8b-instruct': {
		// 	url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
		// 	engine: 'gpt4all',
		// },
	},
}

async function main() {
	const { server, initPromise } = createLLMServer(config)
	server.listen(3000, () => {
		console.debug('Server up')
	})
	initPromise.then(() => {
		console.debug('Pool ready')
	})
}

main().catch((err: Error) => {
	console.error(err)
	process.exit(1)
})

process.on('unhandledRejection', (err) => {
	console.error('Unhandled rejection:', err)
	process.exit(1)
})
