import readline from 'node:readline'
import chalk from 'chalk'
import { ModelServer } from '#lllms/index.js'

// A command-line chat example using the ModelServer.

const llms = new ModelServer({
	// log: 'info',
	models: {
		'dolphin': {
			task: 'text-completion',
			minInstances: 1,
			url: 'https://huggingface.co/QuantFactory/dolphin-2.9-llama3-8b-GGUF/blob/main/dolphin-2.9-llama3-8b.Q4_K_M.gguf',
			engine: 'node-llama-cpp',
			// device: { gpu: false },
		},
	},
})

console.log('Initializing models...')

await llms.start()

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
	process.stdout.write(chalk.bold(chalk.dim('model > ')))
	const result = await llms.processChatCompletionTask(
		{
			model: 'dolphin',
			messages,
		},
		{
			onChunk: (chunk) => {
				process.stdout.write(chunk.text)
			},
		},
	)
	messages.push(result.message)
	process.stdout.write(' ' + chalk.dim(`[${result.finishReason}]`) + '\n')
}
