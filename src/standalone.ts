import type { AddressInfo } from 'node:net'
import { format as formatURL } from 'node:url'
import { startHTTPServer, HTTPServerOptions } from '#lllms/http.js'
import { ChatWithVisionEngine } from '#lllms/lib/custom-engines/ChatWithVision.js'
import { VoiceFunctionCallEngine } from '#lllms/lib/custom-engines/VoiceFunctionCall.js'

// @ts-ignore
import { Florence2ForConditionalGeneration, WhisperForConditionalGeneration } from '@xenova/transformers'

// Currently only used for debugging. Do not use.
const serverOptions: HTTPServerOptions = {
	listen: {
		port: 3000,
	},
	log: 'debug',
	concurrency: 2,
	engines: {
		// 'chat-with-vision': new ChatWithVisionEngine({
		// 	imageToTextModel: 'florence2',
		// 	chatModel: 'llama3-8b',
		// }),
		// 'voice-function-calling': new VoiceFunctionCallEngine({
		// 	speechToTextModel: 'whisper-base',
		// 	chatModel: 'functionary',
		// }),
	},
	models: {
		// 'speech-to-chat': {
		// 	engine: 'speech-to-chat',
		// 	task: 'speech-to-text',
		// },
		
		'whisper-base': {
			url: 'https://huggingface.co/onnx-community/whisper-base',
			engine: 'transformers-js',
			task: 'speech-to-text',
			prepare: 'async',
			minInstances: 1,
			modelClass: WhisperForConditionalGeneration,
			dtype: {
				encoder_model: 'fp32', // 'fp16' works too
				decoder_model_merged: 'q4', // or 'fp32' ('fp16' is broken)
			},
			device: {
				gpu: false,
			},
		},
		'functionary': {
			url: 'https://huggingface.co/meetkai/functionary-small-v2.5-GGUF/raw/main/functionary-small-v2.5.Q4_0.gguf',
			sha256: '3941bf2a5d1381779c60a7ccb39e8c34241e77f918d53c7c61601679b7160c48',
			engine: 'node-llama-cpp',
			task: 'text-completion',
		},
		// 'foo': {
		// 	task:'text-completion2',
		// 	engine: ''
		// }
		// 'foo': {
		// 	'engine': 'transformers-js',
		// 	''
		// },
		// 'llama3-8b': {
		// 	url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
		// 	md5: 'c87ad09e1e4c8f9c35a5fcef52b6f1c9',
		// 	engine: 'gpt4all',
		// 	task: 'text-completion',
		// 	prepare: 'async',
		// },
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
