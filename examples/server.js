import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import express from 'express'
import { LLMServer, createAPIMiddleware } from '../dist/index.js'

// Create a server with a single model, limiting to 2 instances that can run concurrently.
// Models will be downloaded on-demand or during LLMServer.start() if minInstances > 0.
const llmServer = new LLMServer({
  // Default download directory is ~/.cache/lllms.
  // modelsDir: path.resolve(os.homedir(), '.cache/models'),
  inferenceConcurrency: 2,
  models: {
    'phi3-mini-4k': {
      url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
      engine: 'node-llama-cpp',
      maxInstances: 2,
    },
  },
})
llmServer.start()
const app = express()
app.use(
  '/infer',
  express.json(),
  createAPIMiddleware(llmServer),
)
const server = http.createServer(app)
server.listen(3000)

console.log('Server up')

/*
curl http://localhost:3000/infer/openai/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{
			"model": "phi3-mini-4k",
			"messages": [
					{
							"role": "user",
							"content": "how to find my kernel version on linux?"
					}
			]
	}'
*/

/*
{
  "id": "phi3-mini-4k:IZJQCqnA-Euilz5Lu",
  "model": "phi3-mini-4k",
  "object": "chat.completion",
  "created": 1714288309,
  "system_fingerprint": "24c67d07e36aeef1996ba5763bc63db98651ba11",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "To find your kernel version on Linux, you can use several methods. Here are the most common ones:\n\n\n1. Using Terminal:\n\n   - Open a terminal window (you can usually do this by pressing `Ctrl + Alt + T`).\n\n   - Type the following command and press Enter:\n\n     ```\n\n     uname -r\n\n     ```\n\n   This will display your current kernel version.\n\n\n2. Using System Information Tools:\n\n   - Some distributions come with system information tools like `System Monitor` or `gnome-system-monitor`.\n\n   - Open the application and look for an option that displays system details, where you should find the kernel version listed.\n\n\n3. Checking in Software Updates:\n\n   - Some distributions have a software update tool (like `Software Updater` on Ubuntu) which can show your current kernel version under \"System\" or \"About.\"\n\n\n4. Using Command Line with Dmesg:\n\n   - Open the terminal and type:\n\n     ```\n\n     dmesg | grep 'Linux'\n\n     ```\n\n   This command will display messages from the kernel ring buffer, including your kernel version at the top of the output.\n\n\nRemember that if you are using a virtual machine or have multiple kernels installed (like in some distributions), these methods may show different versions depending on which one is currently active."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 310,
    "total_tokens": 320
  }
}
*/