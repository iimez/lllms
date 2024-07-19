import os from 'node:os'
import path from 'node:path'
import chalk from 'chalk'
import { ModelPool } from '#lllms/index.js'
import { elapsedMillis } from '#lllms/lib/util.js'
import * as LlamaCppEngine from '#lllms/engines/node-llama-cpp/engine.js'

// Complete multiple prompts concurrently using ModelPool.

async function onPrepareInstance(instance) {
	// can be used to set up the instance before it's used.
	// the model will not be loaded until this promise resolves.
	// console.log('Instance about to load:', instance)
	// throwing here will put the instance in an error state
}

const pool = new ModelPool(
	{
		// to see what's going on, set the log level to 'debug'
		// log: 'debug',
		// global processing concurrency limit, across all instances of all models
		concurrency: 2,
		models: {
			'phi3-mini-4k': {
				task: 'text-completion',
				// note that this path needs to be absolute and the file needs to be downloaded beforehand.
				location: path.resolve(
					os.homedir(),
					'.cache/lllms/huggingface/bartowski/Phi-3.1-mini-4k-instruct-GGUF-main/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf',
				),
				engine: 'node-llama-cpp',
				minInstances: 1, // setting this to something greater 0 will load the model on pool.init()
				maxInstances: 2, // allow the pool to spawn additional instances of this model
			},
		},
	},
	onPrepareInstance,
)

process.on('exit', () => {
	pool.dispose()
})

console.log('Initializing pool...')
await pool.init({
	'node-llama-cpp': LlamaCppEngine,
})

async function createCompletion(prompt) {
	const req = {
		model: 'phi3-mini-4k',
		prompt,
		temperature: 3,
		maxTokens: 200,
	}
	const completionModel = await pool.requestInstance(req)
	const completionBegin = process.hrtime.bigint()
	const task = completionModel.instance.processTextCompletionTask(req)
	const result = await task.result
	completionModel.release()
	const elapsed = Math.max(elapsedMillis(completionBegin), 1000)
	return {
		text: result.text,
		instance: completionModel.instance.id,
		device: completionModel.instance.gpu ? 'GPU' : 'CPU',
		speed: Math.round(result.completionTokens / (elapsed / 1000)),
	}
}

const printResult = (title) => (result) => {
	console.log(
		chalk.yellow(title) +
			' ' +
			chalk.bold(result.instance) +
			' ' +
			chalk.dim(`generated ${result.speed} tokens/s on ${result.device}`),
	)
	console.log(chalk.dim(prompt) + result.text)
}

const completionCount = 20
const prompt = 'Locality of '

const res = await createCompletion(prompt)
printResult('Initial completion')(res)

console.log(`Processing ${completionCount} completions...`)

for (let i = 1; i <= completionCount; i++) {
	createCompletion(prompt).then(printResult(`#${i}`))
}
