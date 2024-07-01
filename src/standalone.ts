import type { AddressInfo } from 'node:net'
import { format as formatURL } from 'node:url'
import { startHTTPServer, HTTPServerOptions } from '#lllms/http.js'
import { ChatWithVisionEngine } from '#lllms/lib/custom-engines/ChatWithVision.js'

// @ts-ignore
import { Florence2ForConditionalGeneration } from '@xenova/transformers'

// Currently only used for debugging. Do not use.
const serverOptions: HTTPServerOptions = {
	listen: {
		port: 3000,
	},
	log: 'debug',
	concurrency: 2,
	engines: {
		'chat-with-vision': new ChatWithVisionEngine({
			imageToTextModel: 'florence2',
			chatModel: 'llama3-8b',
		}),
	},
	models: {
		'gpt4o': {
			engine: 'chat-with-vision',
			task: 'text-completion',
		},
		florence2: {
			url: 'https://huggingface.co/onnx-community/Florence-2-large-ft',
			engine: 'transformers-js',
			task: 'image-to-text',
			prepare: 'async',
			minInstances: 1,
			engineOptions: {
				modelClass: Florence2ForConditionalGeneration,
				gpu: false,
				dtype: {
					embed_tokens: 'fp16',
					vision_encoder: 'fp32',
					encoder_model: 'fp16',
					decoder_model_merged: 'q4',
				},
			},
		},
		'llama3-8b': {
			url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
			md5: 'c87ad09e1e4c8f9c35a5fcef52b6f1c9',
			engine: 'gpt4all',
			task: 'text-completion',
			prepare: 'async',
		},
		// 'llama3-8b': {
		// 	url: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
		// 	sha256:
		// 		'1977ae6185ef5bc476e27db85bb3d79ca4bd87e7b03399083c297d9c612d334c',
		// 	engine: 'node-llama-cpp',
		// 	task: 'text-completion',
		// 	prepare: 'async',
		// 	engineOptions: {
		// 		gpu: true,
		// 	},
		// },
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
	const server = await startHTTPServer(serverOptions)
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
})
