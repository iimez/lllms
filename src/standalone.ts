import { createServer, InferenceServerOptions } from './server.js'

const config: InferenceServerOptions = {
	concurrency: 1,
	models: {
		'orca:3b': {
			url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
			preload: true,
			engine: 'gpt4all',
		},
		'llama3:8b': {
			url: 'https://huggingface.co/NousResearch/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B-Q4_K_M.gguf',
			preload: true,
			engine: 'node-llama-cpp',
		},
	},
}

async function main() {
	const {server, initPromise } = await createServer(config)
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
