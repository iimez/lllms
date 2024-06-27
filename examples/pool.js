import os from 'node:os'
import path from 'node:path'
import chalk from 'chalk'
import { LLMPool } from '../dist/index.js'
import { elapsedMillis } from '../dist/lib/util.js'
import { createLogger } from '../dist/lib/logger.js'

// Complete multiple prompts concurrently using LLMPool.

async function onPrepareInstance(instance) {
	// can be used to set up the instance before it's used.
	// the model will not be loaded until this promise resolves.
	// console.log('Instance about to load:', instance)
	// throwing here will put the instance in an error state
}

const pool = new LLMPool(
	{
		// to see what's going on, set the log level to 'debug'
		log: 'debug',
		// global processing concurrency limit, across all instances of all models
		concurrency: 2,
		models: {
			'phi3-mini-4k': {
				// note that this path needs to be absolute and the file needs to be downloaded beforehand.
				file: path.resolve(
					os.homedir(),
					'.cache/lllms/Phi-3-mini-4k-instruct.Q4_0.gguf',
				),
				engine: 'gpt4all',
				minInstances: 1, // setting this to something greater 0 will load the model on pool.init()
				maxInstances: 2, // allow the pool to spawn additional instances of this model
			},
		},
	},
	onPrepareInstance,
)

console.log('Initializing pool...')
await pool.init()

async function createCompletion(prompt) {
	const req = {
		model: 'phi3-mini-4k',
		prompt,
		temperature: 3,
		maxTokens: 200,
	}
	const { instance, release } = await pool.requestInstance(req)
	const completion = instance.createCompletion(req)
	const completionBegin = process.hrtime.bigint()
	const result = await completion.process()
	release()
	const elapsed = Math.max(elapsedMillis(completionBegin), 1000)
	return {
		text: result.text,
		instance: instance.id,
		device: instance.gpu ? 'GPU' : 'CPU',
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
