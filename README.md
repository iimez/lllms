## lllms

Local Large Language Models. Providing an LLM instance pool and tools to run and host multiple models on any machine. Built on [llama.cpp](https://github.com/ggerganov/llama.cpp/) via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) and [gpt4all](https://github.com/nomic-ai/gpt4all).

Note that this is a dev and learning tool and not meant for production use. It is not secure, not scalable, and (currently) not (enough) optimized for performance. It is meant to be used on a local machine or a small server for personal use or small scale experiments. Prioritizing ease of use and simple APIs. For larger scale hosting see [these other solutions](#related-solutions).

⚠️ This package is currently in beta. Some APIs may change. Feel free to report any issues you encounter.

### Features

- Run multiple large language models concurrently, adjust cache lifetimes and 
- OpenAI spec API endpoints. See [#progress](#progress) for compatibility.
- Cache loaded models and their contexts/sessions across stateless API requests.
- BYO web server or use the provided express server and middleware.
- Or use the LLM instance pool directly within your application.
- Automatically downloads and caches GGUF's to `~/.cache/lllms`.

### Usage

#### JavaScript APIs

To integrate lllms directly with your application, you can use either the higher level `startLLMs` or the lower level `LLMPool`. For the latter check out [./examples/pool](./examples/pool.js) and [./examples/cli-chat](./examples/cli-chat.js)

To attach lllms to your existing (express, or any other) web server see [./examples/express-openai](./examples/express-openai.js) and [./examples/server-node](./examples/server-node.js). See [./src/http.ts](./src/http.ts) for more ways to integrate with existing HTTP servers.


```js lllms.js
import { serveLLMs } from 'lllms'

// Starts a http server for up to two instances of phi3 and exposes them via openai API
serveLLMs({
  // Limit how many completions can be processed concurrently. If its exceeded, requests
  // will stall until a slot is free. Defaults to 1.
  inferenceConcurrency: 2,
  downloadConcurrency: 2, // How many models may be downloaded concurrently
  // Where to write models to, defaults to `~/.cache/lllms`
  // modelsPath: '/path/to/models',
  models: {
    // Specify as many models as you want. Identifiers can use a-zA-Z0-9_:\-
    // Only URL is required. Per default models will only be downloaded on demand.
    'phi3-mini-4k': {
      // Model weights may be specified by file and/or url.
      url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
      // Absolute or relative to modelsPath. If it does not exist and a url is configured
      // it will be downloaded to the given location.
      file: 'Phi-3-mini-4k-instruct-q4.gguf',
      // Checksums are optional and will be verified before loading the model if set.
      // md5: 'cb68b653b24dc432b78b02df76921a54',
      // sha256: '8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef',
      // Use these to control how much memory your LLMs may use.
      contextSize: 4096, // Maximum context size. Will be determined automatically if not set.
      maxInstances: 2, // How many active sessions you wanna be able to cache at the same time.
      minInstances: 1, // To always keep at least one instance ready.
      ttl: 300, // Idle sessions will be disposed after this many seconds.
      // Set defaults for completions. These can be overridden per request.
      // If unset, default values depend on the engine.
      systemPrompt: 'You are a helpful assistant.',
      completionDefaults: {
        temperature: 1,
      //   maxTokens: 100,
      //   seed: 1234,
      //   stop: ['\n'],
      //   repeatPenalty: 0.6,
      //   repeatPenaltyNum: 64,
      //   frequencyPenalty: 0,
      //   presencePenalty: 0,
      //   topP: 1,
      //   minP: 0,
      //   topK: 0,
      //   grammar: 'json',
      //   tokenBias: {
      //     'no': 100,
      //     'yes': -100,
      //   },
      },
      // Choose between node-llama-cpp or gpt4all as bindings to llama.cpp.
      engine: 'node-llama-cpp',
      engineOptions: {
        // GPU will be used automatically, but models can be forced to always run on gpu by
        // setting to true. Note that both engines currently do not support running multiple
        // models on gpu at the same time. Requests will stall until a gpu slot is free and
        // context cannot be reused if there are multiple sessions going on.
        // gpu: true,
        // batchSize: 512,
        cpuThreads: 4,
        // memLock: true, // Only supported for node-llama-cpp.
      },
    },
  },
  // HTTP listen options. If you don't need a web server, use startLLMs or `new LLMServer()`
  // directly instead of `serveLLMs`. Apart from `listen` they take the same configuration.
  listen: {
    port: 3000,
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

- `GET /` - Prints info about spawned instances, available models and ongoing downloads.

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
| logit_bias          | ❌      | ✅             |
| response_format     | ❌      | ✅             |
| tools               | ❌      | ✅             |
| tool_choice         | ❌      | ❌             |
| suffix              | ❌      | ❌             |
| echo                | ❌      | ❌             |

| Additional params   | gpt4all | node-llama-cpp |
| ------------------- | ------- | -------------- |
| top_k               | ✅      | ✅             |
| min_p               | ✅      | ✅             |
| repeat_penalty_num  | ✅      | ✅             |
| repeat_penalty      | ✅      | -              |

#### Functionality

| Feature               | gpt4all | node-llama-cpp |
| --------------------- | ------- | -------------- |
| Context cache         | ✅      | ✅             |
| System prompt         | ✅      | ✅             |
| GPU                   | ✅      | ✅             |
| Content part messages | ❌      | ❌             |
| Function Calling      | ❌      | ✅             |


#### Limitations and Known Issues

##### System Messages
System role messages are supported only as the first message in a chat completion session. All other system messages will be ignored.

##### Context Cache
Note that the current context cache implementation for the OpenAI API only works if (apart from the final user message) the _same messages_ are resent in the _same order_. This is because the messages will be hashed to be compared during follow up turns, to match requests to the correct session. If no hash matches everything will still work, but slower. Because a fresh context will be used and passed messages will be reingested.

##### Function Calling
Parallel function calls are currently not possible. `tool_choice` will always be `auto`.

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
- [x] Allow configuring model hashes / verification
- [x] Improve template code / stop trigger support
- [x] Support configuring a timeout on completion processing
- [x] Logit bias / Token bias support
- [ ] Improve node-llama-cpp token usage counts / TokenMeter
- [ ] Support preloading instances with context, like a long system message or few shot examples
- [ ] Improve tests for longer conversations / context window shifting
- [ ] Tests for request cancellation
- [ ] A mock engine implementing for testing and examples / Allow user to customize engine implementations
- [ ] Embeddings APIs
- [ ] Logprobs support
- [ ] Replace express with tinyhttp?
- [ ] Script to generate a minimal dummy/testing GGUF https://github.com/ggerganov/llama.cpp/discussions/5038#discussioncomment-8181056
- [ ] Allow configuring total RAM/VRAM usage (See https://github.com/ggerganov/llama.cpp/issues/4315 Check estimates?)

### Contributing

If you know how to fill in any of the above checkboxes or have additional ideas you'd like to make happen, feel free to open an issue or PR.

#### Possible Future Goals

- Create a separate HTTP API thats independent of the OpenAI spec.
- Add a clientside library (React hooks?) for use of this independent API.
- Provide a CLI. (Launch a server via `lllms serve config.json|js`? Something to manage models on disk?)
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