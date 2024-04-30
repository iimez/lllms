import http from 'node:http'
import express from 'express'
import { LLMServer, createExpressMiddleware } from '../dist/index.js'

// Demonstration of using the LLMServer + Express middleware to serve an OpenAI API.

// Create a server with a single model, limiting to 2 instances that can run concurrently.
// Models will be downloaded on-demand or during LLMServer.start() if minInstances > 0.
const llms = new LLMServer({
  // Default download directory is ~/.cache/lllms.
  // modelsDir: path.resolve(os.homedir(), '.cache/models'),
  concurrency: 2,
  models: {
    'phi3-mini-4k': {
      url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
      engine: 'node-llama-cpp',
      maxInstances: 2,
    },
  },
})
llms.start()
const app = express()
app.use(
  express.json(),
  createExpressMiddleware(llms),
)
const server = http.createServer(app)
server.listen(3000)

console.log('Server up')

/*
curl http://localhost:3000/openai/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{
			"model": "phi3-mini-4k",
			"messages": [
					{
							"role": "user",
							"content": "Lets count to three!"
					}
			]
	}'
*/

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