import os from 'node:os'
import path from 'node:path'
import chalk from 'chalk'
import { LLMPool } from '../dist/index.js'

// Complete multiple prompts concurrently using LLMPool.

async function onPrepareInstance(instance) {
	// can be used to set up the instance before it's used.
	// the model will not be loaded until this promise resolves.
	// console.log('Instance about to load:', instance)
}

const pool = new LLMPool(
	{
		// global inference concurrency limit, across all models
		concurrency: 2,
		models: {
			'phi3-mini-4k': {
				// note that this file needs to be downloaded manually when using the pool directly.
				file: path.resolve(
					os.homedir(),
					'.cache/lllms/Phi-3-mini-4k-instruct.Q4_0.gguf',
				),
				engine: 'gpt4all',
				// setting this to 1 will load the model on pool.init(), otherwise it will be loaded on-demand
				minInstances: 1,
				maxInstances: 2, // allow the pool to spawn additional instances of this model
			},
		},
	},
	onPrepareInstance,
)

console.log('Initializing pool...')
await pool.init()

console.log('Pool ready')

async function createCompletion(prompt) {
	const req = {
		model: 'phi3-mini-4k',
		prompt,
		temperature: 3,
		maxTokens: 50,
	}
	const { instance, releaseInstance } = await pool.requestCompletionInstance(
		req,
	)
	const completion = instance.createCompletion(req)

	const completionBegin = process.hrtime()
	const result = await completion.process()
	const elapsed = process.hrtime(completionBegin)

	releaseInstance()
	return {
		text: result.text,
		instance: instance.id,
		speed: Math.round(req.maxTokens / (elapsed[0] + elapsed[1] / 1e9)),
	}
}

const printResult = (title) => (result) => {
	console.log(
		chalk.yellow(title) +
			' ' +
			chalk.bold(result.instance) +
			' ' +
			chalk.dim(`generated ${result.speed} tokens/s`),
	)
	console.log(chalk.dim(prompt) + result.text)
}

console.log('Processing completions...')

const prompt = 'Locality of '
const res = await createCompletion(prompt)
printResult('Solo completion')(res)

const count = 5

for (let i = 1; i <= count; i++) {
	createCompletion(prompt).then(printResult(`#${i}`))
}
process.nextTick(() => {
	const { waiting, processing } = pool.getStatus()
	console.log('Pool processing', { waiting, processing })
})