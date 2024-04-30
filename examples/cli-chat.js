import os from 'node:os'
import path from 'node:path'
import readline from 'node:readline'
import chalk from 'chalk'
import { LLMPool } from '../dist/index.js'

// A simple command-line chat example using the LLMPool class.
// Can use LLMPool instead of LLMServer if you wanna manage your model files manually.

const pool = new LLMPool(
	{
		// default concurrency=1, which is okay for this example.
		// concurrency: 2,
		models: {
			'phi3-mini-4k': {
				// note that this file needs to be downloaded manually when using the pool directly.
				file: path.resolve(os.homedir(), '.cache/lllms/Phi-3-mini-4k-instruct.Q4_0.gguf'),
				engine: 'gpt4all',
				// setting this to 1 will load the model on pool.init(), otherwise it will be loaded on-demand
				// minInstances: 1,
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
	const { instance, releaseInstance } = await pool.requestCompletionInstance(req)
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
	releaseInstance()
}
