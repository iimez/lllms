## lllms

Local Large Language Models. Providing an LLM instance pool and tools to run and host multiple large language models on any machine. Built on [llama.cpp](https://github.com/ggerganov/llama.cpp/) via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) and [gpt4all](https://github.com/nomic-ai/gpt4all).

⚠️ This package is currently just a draft/experiment and not ready for consumption.

### Goals

- Allow configuring GGUF's and automatically download them.
- Abstract away the underlying "engine" - implement engine interface for both gpt4all and node-llama-cpp.
- Support running multiple models concurrently and allow configuration specific to use case and hardware.
- Support preloading models, preparing them for incoming requests. Or lazy loading them only on demand.
- Provide OpenAI API compatible HTTP endpoints.
- Cache loaded models and their contexts/sessions across requests, even when using the stateless API.
- Allow usage of the package as (a) library (no http involved) or (b) standalone server or (c) node http request handler.

#### Possible Future Goals

- Create a separate HTTP API thats independent of the OpenAI spec.
- Add a clientside library for use of this independent HTTP API.
- Provide a CLI.
- Provide a Docker image.

#### Currently not the Goals

- Something that is production ready or scalable. Theres more appropriate tools/servies if you wanna host open models at scale.
- Another facade to LLM hoster HTTP API's. This is for local/private/offline use.
- Worry about authentication or rate limiting or misuse. Host this responsibly.
- Any kind of distributed or multi-node setup. This is not scope for this project.

### Progress

#### OpenAI API

| OpenAI API Feature  | gpt4all | node-llama-cpp |
| ------------------- | ------- | -------------- |
| v1/chat/completions | ✅      | ✅             |
| v1/completions      | ✅      | ✅             |
| Streaming           | ✅      | ✅             |
| v1/embeddings       | ❌      | ❌             |
| v1/models           | 🚧      |

### Use as a library

All APIs subject to change.

```ts
import { LLMPool } from 'lllms'
const pool = new LLMPool({
  concurrency: 2,
  models: {
    'phi3-4k': {
      url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
      minInstances: 1,
      maxInstances: 3,
      engine: 'node-llama-cpp',
    },
    'orca-3b': {
      url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
      minInstances: 1,
      engine: 'gpt4all',
    },
  },
})
const server = http.createServer(async (req, res) => {
  try {
    const { messages, model } = JSON.parse(req.body)
    const lock = await pool.requestCompletionInstance({ model, messages })
    const completion = lock.instance.createChatCompletion({ messages })
    const result = await completion.processChatCompletion(request)
    lock.release()
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(result, null, 2))
  } catch (e) {
    console.error(e)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Internal server error' }))
  }
})
pool.init()
server.listen(3000)
```

### Use server with OpenAI API Endpoints

See [./src/standalone.ts](./src/standalone.ts) for an example.

```bash
npm install
npm run build
npm run start
```

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "llama3-8b",
      "messages": [
          {
              "role": "user",
              "content": "Whats 1+1?"
          }
      ]
  }'
```

```bash
curl http://localhost:3000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "phi3-4k",
      "prompt": "To verify our code works we will first write an integration",
      "temperature": 0,
      "max_tokens": 1
  }'
```

### Related Solutions

If you look at this package, you probably also want to take a look at these other solutions:

- [ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) - Uses llama.cpp and provides a HTTP API. Also has experimental OpenAI API compatibility.
- [llama.cpp Server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#llamacpp-http-server) - The official llama.cpp HTTP API.
- [VLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) - A more production ready solution for hosting large language models.