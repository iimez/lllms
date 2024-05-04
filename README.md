## lllms

Local Large Language Models. Providing an LLM instance pool and tools to run and host multiple models on any machine. Built on [llama.cpp](https://github.com/ggerganov/llama.cpp/) via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) and [gpt4all](https://github.com/nomic-ai/gpt4all).

Note that this is a dev and learning tool and not meant for production use. It is not secure, not scalable, and (currently) not (enough) optimized for performance. It is meant to be used on a local machine or a small server for personal use or small scale experiments. Prioritizing ease of use and simple APIs. For larger scale hosting see [these other solutions](#related-solutions).

⚠️ This package is currently a WIP and many things are subject to change or not yet implemented.

### Features

- Run multiple large language models concurrently, until your CPU or RAM runs out.
- Automatically download and cache GGUF's to `~/.cache/lllms`.
- OpenAI spec API endpoints. See [#progress](#progress) for compatibility.
- Cache loaded models and their contexts/sessions across requests.
- BYO web server or use the provided express server and middleware.
- Use the LLM instance pool on its own within your application.

### Usage

#### JavaScript APIs

To integrate lllms directly with your application, you can use either the higher level `startLLMs` or the lower level `LLMPool`. For the latter check out [./examples/pool](./examples/pool.js) and [./examples/cli-chat](./examples/cli-chat.js)

To attach lllms to your existing (express, or any other) web server see [./examples/express-openai](./examples/express-openai.js) and [./examples/server-node](./examples/server-node.js). See [./src/server.ts](./src/server.ts) for more ways to integrate with existing HTTP servers.

The highest level API, to spin up a standalone server:

```js lllms.js
import { serveLLMs } from 'lllms'

// Starts a http server for up to two instances of phi3 and exposes them via openai API
serveLLMs({
  // Web server options. If you dont need a web server, use startLLMs or
  // `new LLMServer()` instead. Apart from `listen` they take the same configuration.
  listen: {
    port: 3000,
  },
  // Limit how many completions can be processed concurrently. If its exceeded, requests
  // will stall until a slot is free. Only relevant for multiple cpu instances, as only
  // one gpu instance can run at a time.
  concurrency: 2,
  // Custom model directory, defaults to `~/.cache/lllms`
  // modelsDir: '/path/to/models',
  models: {
    // Model names can use a-zA-Z0-9_:\-
    'phi3-mini-4k': {
      // Model weights may be specified by file or url.
      url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
      file: 'Phi-3-mini-4k-instruct-q4.gguf',
      // Choose between node-llama-cpp or gpt4all as bindings to llama.cpp.
      engine: 'node-llama-cpp',
      engineOptions: {
        // GPU will be enabled automatically, but models can be forced to always run on gpu by
        // setting to true. Note that both engines currently do not support running multiple
        // models on gpu at the same time. Requests will stall until a gpu slot is free and
        // context cannot be reused if theres multiple sessions going on.
        gpu: true,
        batchSize: 512,
        cpuThreads: 4,
        memLock: true, // Only supported for node-llama-cpp.
      },
      // Maximum context size. Will be determined automatically if not set.
      contextSize: 4096,
      maxInstances: 2, // Set this to how many sessions you expect to run concurrently.
      minInstances: 1, // To download on startup and immediately prepare an instance.
      ttl: 300, // Model instance time to live in seconds.
    },
  },
})
// During download completion requests will stall to get processed once the model is ready.
// You can navigate to http://localhost:3000 to view status.
```
```sh
$ curl http://localhost:3000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "phi3-mini-4k",
      "messages": [
          {
              "role": "user",
              "content": "lets count to 10"
          }
      ]
  }'
```
```json
{
  "id": "phi3-mini-4k:z5SBqZhf-jBO59uI8",
  "model": "phi3-mini-4k",
  "object": "chat.completion",
  "created": 1714292409,
  "system_fingerprint": "0159c68a067a360e4be3e285d3e309440c070734",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Sure, let's count to 10 together:\n\n1. One\n2. Two\n3. Three\n4. Four\n5. Five\n6. Six\n7. Seven\n8. Eight\n9. Nine\n10. Ten\n\nWe have now reached the number 10!"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 69,
    "total_tokens": 75
  }
}
```

#### HTTP API

On the packaged server there is only one additional HTTP endpoint that is not part of the OpenAI API at `/openai/v1`.

- `GET /` - Prints the pool status and download stats.

### Progress

#### OpenAI API Support

| Endpoints           | gpt4all | node-llama-cpp |
| ------------------- | ------- | -------------- |
| v1/chat/completions | ✅      | ✅             |
| v1/completions      | ✅      | ✅             |
| v1/embeddings       | ❌      | ❌             |
| v1/models           | 🚧      |

| Spec params         | gpt4all | node-llama-cpp |
| ------------------- | ------- | -------------- |
| stream              | ✅      | ✅             |
| temperature         | ✅      | ✅             |
| max_tokens          | ✅      | ✅             |
| top_p               | ✅      | ✅             |
| stop                | ✅      | ✅             |
| seed                | ❌      | ✅             |
| frequency_penalty   | ❌      | ✅             |
| presence_penalty    | ❌      | ✅             |
| best_of             | ❌      | ❌             |
| n                   | ❌      | ❌             |
| logprobs            | ❌      | ❌             |
| top_logprobs        | ❌      | ❌             |
| logit_bias          | ❌      | ❌             |
| response_format     | ❌      | ❌             |
| tools               | ❌      | ❌             |
| tool_choice         | ❌      | ❌             |
| suffix              | ❌      | ❌             |
| echo                | ❌      | ❌             |

| Additional params   | gpt4all | node-llama-cpp |
| ------------------- | ------- | -------------- |
| top_k               | ✅      | ✅             |
| min_p               | ✅      | ✅             |
| repeat_penalty_num  | ✅      | -              |
| repeat_penalty      | ✅      | -              |

#### Functionality

| Feature               | gpt4all | node-llama-cpp |
| --------------------- | ------- | -------------- |
| Context cache         | ✅      | ✅             |
| System prompt         | ✅      | ✅             |
| GPU                   | ✅      | ✅             |
| Content part messages | ❌      | ❌             |
| Function Calling      | ❌      | ❌             |


System role messages are supported only as the first message in a chat completion session. All other system messages will be ignored.

Note that the current context-reuse implementation only works if (apart from the final user message) the same messages are resent in the same order. This is because the messages will be hashed to be compared during follow up turns. If no hash matches, a fresh context will be used and passed messages will be reingested.

#### TODO / Roadmap

Not in any particular order:

- [x] Automatic download of GGUF's with ipull
- [x] Engine abstraction
- [x] Model instance pool and queue
- [x] Basic OpenAI API compatibility
- [x] POC of chat context reuse across requests
- [x] Tests for context reuse and context leaking
- [x] Logging Interface
- [x] Better Examples
- [x] GPU support
- [x] node-llama-cpp context reuse
- [x] Instance TTL
- [ ] Tests for longer conversations
- [ ] Tests for request cancellation
- [ ] Better template configuration options
- [ ] Expose more download configuration options
- [ ] Allow configuring model hashes
- [ ] Support preloading session contexts, like a long system message or few shot examples
- [ ] Allow configuration to limit RAM usage
- [ ] Support configuring a timeout on completion processing
- [ ] CLI to spin up a server given a config file or a model name
- [ ] API to add custom engines
- [ ] A mock engine for testing and implementing old-school chatbots
- [ ] Support Embeddings API

### Contributing

If you know how to fill in any of the above checkboxes or have additional ideas you'd like to make happen, feel free to open an issue or PR.

#### Possible Future Goals

- Create a separate HTTP API thats independent of the OpenAI spec.
- Add a clientside library (React hook?) for use of this independent API.
- Provide a CLI. (Launch a server via `lllms serve config.json|js`?)
- Provide a Docker image. And maybe a Prometheus endpoint.

#### Currently not the Goals

- Another facade to LLM hoster HTTP API's. The strengths here are local/private/offline use.
- Worry about authentication or rate limiting or misuse. Host this with caution, or build these things on top.
- Some kind of distributed or multi-node setup. Does not fit the scope well.
- Other tooling like vector stores, Chat GUIs, etc. Would wish for this to stay minimal and easy to understand.

### Related Solutions

If you look at this package, you might also want to take a look at these other solutions:

- [ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) - Uses llama.cpp and provides a HTTP API. Also has experimental OpenAI API compatibility.
- [llama.cpp Server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#llamacpp-http-server) - The official llama.cpp HTTP API.
- [VLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) - A more production ready solution for hosting large language models.
- [LM Studio](https://lmstudio.ai/docs/local-server) - Also has a local server.