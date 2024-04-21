## lllms

Local Large Language Models. Providing a toolkit to run and host multiple large language models on any machine. Built on [llama.cpp](https://github.com/ggerganov/llama.cpp/) via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) and [gpt4all](https://github.com/nomic-ai/gpt4all).

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

| OpenAI API Feature   | gpt4all | node-llama-cpp |
|----------------------|---------|----------------|
| v1/chat/completions  |   🚧    |       🚧       |
| v1/completions       |   ❌    |       ❌       |
| Streaming            |   ❌    |       ❌       |

### Use as a library

All APIs subject to change.

```ts
import { LLMPool } from 'inference-server'
const pool = new LLMPool({
  concurrency: 1,
  models: {
    'orca:3b': {
      url: 'https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf',
      preload: true,
      engine: 'gpt4all',
    },
    'llama3:8b': {
      url: 'https://huggingface.co/NousResearch/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B-Q4_K_M.gguf',
      preload: true,
      engine: 'node-llama-cpp',
    },
  },
})
const server = http.createServer((req, res) => {
  try {
    const messages = JSON.parse(req.body)
    const { instance, request } = await pool.requestChatCompletionInstance({
      model,
      messages,
    })
    const completion = await instance.processChatCompletion(request)
    instance.unlock()
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(completion, null, 2))
  } catch (e) {
    console.error(e)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Internal server error' }))
  }
})
pool.init()
server.listen(3000)
```

### Run standalone

```bash
npm install
npm run build
npm run start
```

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "llama3:8b",
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
    "model": "gpt-3.5-turbo-instruct",
    "prompt": "Say this is a test",
    "max_tokens": 7,
    "temperature": 0
  }'
```

### Related Solutions

If you look at this package, you should also look at these other solutions:
- [ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) - Uses llama.cpp and provides a HTTP API. Also has experimental OpenAI API compatibility.
- [llama.cpp Server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#llamacpp-http-server) - The official llama.cpp HTTP API.