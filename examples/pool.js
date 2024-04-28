import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import express from 'express'
import { LLMPool } from '../dist/index.js'

// pool requires all models to have an absolute path configured
const modelsDir = path.resolve(os.homedir(), '.cache/lllms')

// optional callback for custom initialization logic, used by LLMServer to download models on demand.
const prepareInstance = (instance) => {
	console.debug('Pool wants to start instance', instance)
	return Promise.resolve()
}

// create our pool
const pool = new LLMPool({
	inferenceConcurrency: 2,
	models: {
		'phi3-mini-4k': {
			// pool will error if this file does not exist
			file: path.join(modelsDir, 'Phi-3-mini-4k-instruct-q4.gguf'),
			engine: 'node-llama-cpp',
			maxInstances: 2,
		},
	},
}, prepareInstance)
pool.init()

// start a web server and add an endpoint for chat
const app = express()
app.use(express.json())
app.set('json spaces', 2)
app.use('/chat', async (req, res) => {
	const { model, messages, temperature, maxTokens } = req.body
	const { instance, releaseInstance } = await pool.requestCompletionInstance({
		model,
		messages,
	})
	const completion = instance.createChatCompletion({
		model,
		messages,
		temperature,
		maxTokens,
	})
	const result = await completion.process()
	releaseInstance()
	res.json(result)
})
const httpServer = http.createServer(app)
httpServer.listen(3000)

/*
curl http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
      "model": "phi3-mini-4k",
      "messages": [
          {
              "role": "user",
              "content": "how to find my kernel version on linux=?"
          }
      ]
  }'
*/

/*
{
  "finishReason": "eogToken",
  "message": {
    "role": "assistant",
    "content": "To find your kernel version on Linux, you can use the following methods: [...]"
  },
  "promptTokens": 10,
  "completionTokens": 344,
  "totalTokens": 354
}
*/
