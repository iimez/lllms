import os from 'node:os'
import path from 'node:path'
import readline from 'node:readline'
import chalk from 'chalk'
import { LLMPool } from '../dist/index.js'

// A simple command-line chat example using the LLMPool class.
// Can use LLMPool instead of LLMServer if you wanna manage your model files manually.

const pool = new LLMPool(
	{
		// log: 'debug',
		models: {
			'phi3-mini-4k': {
				task: 'text-completion',
				prepare: 'blocking',
				minInstances: 1,
				// note that this file needs to be downloaded manually when using the pool directly.
				// setting this to 1 will load the model on pool.init(), otherwise it will be loaded on-demand
				file: path.resolve(
					os.homedir(),
					// '.cache/lllms/huggingface/microsoft/Phi-3-mini-4k-instruct-gguf-main/Phi-3-mini-4k-instruct-q4.gguf',
					'.cache/lllms/huggingface/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-main/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
					// '.cache/lllms/huggingface/QuantFactory/Phi-3-mini-128k-instruct-GGUF-main/Phi-3-mini-128k-instruct.Q4_0.gguf',
				),
				engine: 'node-llama-cpp',
				// engineOptions: {
				// 	gpu: false,
				// },
			},
		},
	}
)

console.log('Initializing pool...')
await pool.init()

const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
})

const messages = []

while (true) {
	const input = await new Promise((resolve) => {
		rl.question(chalk.bold(chalk.dim('user > ')), (input) => {
			resolve(input)
		})
	})
	messages.push({
		role: 'user',
		content: input,
	})
	// note that this will be used ...
	const req = {
		model: 'phi3-mini-4k',
		messages,
	}
	// ... both to decide which instance to use ...
	const { instance, release } = await pool.requestInstance(req)
	// ... and to create the completion
	const completion = instance.createChatCompletion(req)
	process.stdout.write(chalk.bold(chalk.dim('model > ')))
	const result = await completion.process({
		onChunk: (chunk) => {
			process.stdout.write(chunk.text)
		},
	})
	
	messages.push(result.message)
	process.stdout.write(' ' + chalk.dim(`[${result.finishReason}]`) + '\n')
	console.debug({
		promptTokens: result.promptTokens,
		completionTokens: result.completionTokens,
		totalTokens: result.totalTokens,
	})
	// don't forget to release the instance, or the followup turn will be blocked
	await release()
}
