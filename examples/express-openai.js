import http from 'node:http'
import express from 'express'
import OpenAI from 'openai'
import { LLMServer } from '../dist/server.js'
import { createExpressMiddleware } from '../dist/http.js'

// Demonstration of using the LLMServer + Express middleware to serve an OpenAI API.

// Create a server with a single model, limiting to 2 instances that can run concurrently.
// Models will be downloaded on-demand or during LLMServer.start() if minInstances > 0.
const llms = new LLMServer({
	// Default model path is ~/.cache/lllms.
	// modelsPath: path.resolve(os.homedir(), '.cache/models'),
	concurrency: 2,
	models: {
		'phi3-mini-4k': {
			task: 'text-completion',
			url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
			engine: 'node-llama-cpp',
			maxInstances: 2,
		},
	},
})

await llms.start()

const app = express()
app.use(express.json(), createExpressMiddleware(llms))
const server = http.createServer(app)
server.listen(3001)

console.log('Server up, sending chat completion request...')

const openai = new OpenAI({
	baseURL: 'http://localhost:3001/openai/v1/',
	apiKey: '123',
})

const completion = await openai.chat.completions.create({
	model: 'phi3-mini-4k',
	messages: [{ role: 'user', content: 'Lets count to three!' }],
	stop: ['Two'],
})

console.log(JSON.stringify(completion, null, 2))

/*
{
  "id": "phi3-mini-4k:pU2BHWUv-kHdAeVn8",
  "model": "phi3-mini-4k",
  "object": "chat.completion",
  "created": 1714431837,
  "system_fingerprint": "0159c68a067a360e4be3e285d3e309440c070734",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Sure, let's count together: 1 (one), 2 (two), and 3 (three). If you have any other questions or need further assistance, feel free to ask!"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 41,
    "total_tokens": 47
  }
}
*/
