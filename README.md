## lllms

Local Large Language Models. Providing an instance pool and tools to run your LLM application on any machine. Built on [llama.cpp](https://github.com/ggerganov/llama.cpp/) via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) and [gpt4all](https://github.com/nomic-ai/gpt4all).

Power LLM applications on localhost. Host chatbots for a few users on minimal hardware. Prioritizing ease of use and simple APIs. For larger scale hosting see [these other solutions](#related-solutions). 

Note that this is currently not secure (ie the HTTP API is probably DoS-able, only minimal input validation) and not designed to be scalable beyond a single machine. Misconfiguration or misuse can lead to process crashes or high memory usage. There are no additional safeguards in place and node.js memory limits will not apply.

⚠️ This package is currently in beta. Some APIs may change. Feel free to report any issues you encounter.

### Features

- Configure as many models as you want, download and cache them on demand to `~/.cache/lllms`. Or provide them as abs file paths.
- Adjust the pools `concurrency`, and the models `maxInstances`, `ttl` and `contextSize` to fit your usecase. Can be tuned to either use no resources when idle or to always keep a model ready.
- OpenAI spec API endpoints. See [#progress](#progress) for compatibility. Including a chat session cache that attempts to reuse existing sessions across stateless api requests.
- BYO web server or use the provided express server and middleware.
- Or directly use the LLM instance pool within your application.

### Usage

#### JavaScript APIs

To integrate lllms directly with your application, you can use either the higher level `startLLMs` or the lower level `LLMPool`. For the latter check out [./examples/pool](./examples/pool.js) and [./examples/cli-chat](./examples/cli-chat.js)

To attach lllms to your existing (express, or any other) web server see [./examples/express-openai](./examples/express-openai.js) and [./examples/server-node](./examples/server-node.js). See [./src/http.ts](./src/http.ts) for more ways to integrate with existing HTTP servers.


```js lllms.js
import { serveLLMs } from 'lllms'

// Starts a http server for up to two instances of phi3 and exposes them via openai API
serveLLMs({
  // Limit how many instances can be used concurrently. If its exceeded, requests
  // will stall until a model instance is released. Defaults to 1.
  concurrency: 2,
  // Where to cache models to. Defaults to `~/.cache/lllms`
  // modelsPath: '/path/to/models',
  models: {
    // Specify as many models as you want. Identifiers can use a-zA-Z0-9_:\-
    // Required are `task`, `engine`, `url` and/or `file`.
    'phi3-mini-4k': {
      task: 'inference', // Use 'inference' or 'embedding'
      engine: 'node-llama-cpp', // Choose between node-llama-cpp or gpt4all as bindings.
      // Model weights may be specified by file and/or url.
      url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
      // Abs file path or relative to modelsPath. If it does not exist and a url is configured
      // it will be downloaded to the given location.
      file: 'Phi-3-mini-4k-instruct-q4.gguf',
      // Checksums are optional and will be verified when preparing the model, if set.
      // md5: 'cb68b653b24dc432b78b02df76921a54',
      // sha256: '8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef',
      // When to download and verify models weights. Defaults to 'on-demand'.
      // "blocking" = download on startup, "async" = download on startup but don't block.
      prepare: 'on-demand',
      // Whether a chat session should be preloaded when a new instance is created
      // currently only takes "chat" as a value, which will ingest system prompt and function call docs.
      preload: 'chat',
      // Use these to control resource usage.
      contextSize: 4096, // Maximum context size. Will be determined automatically if not set.
      maxInstances: 2, // How many active sessions you wanna be able to cache at the same time.
      minInstances: 1, // To always keep at least one instance ready. Defaults to 0.
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
      // Available options and their defaults are engine specific.
      engineOptions: {
        // GPU will be used automatically, but models can be forced to always run on gpu by
        // setting `gpu` to true. Usually better to leave it unset. See limitations.
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
| v1/embeddings       | ✅      | ✅             |
| v1/models           | ✅      | ✅             |

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
| Chat context cache    | ✅      | ✅             |
| System prompt         | ✅      | ✅             |
| GPU                   | ✅      | ✅             |
| Content part messages | ❌      | ❌             |
| Grammar               | ❌      | ✅             |
| Function Calling      | ❌      | ✅             |


#### Limitations and Known Issues

##### GPU Support limited to one model
Both engines currently do not support running multiple models on gpu at the same time. This means if gpu is force enabled for a model requests will stall until a gpu slot is free. Otherwise more cpu instances will be spawned. And gpu will be used by the first instance that requests it.
If you require closer control over which requests exactly use gpu, you can always configure a model multiple times with different parameters.

##### System Messages
System role messages are supported only as the first message in a chat completion session. All other system messages will be ignored.

##### Chat Context Cache
Note that the current context cache implementation only works if (apart from the final user message) the _same messages_ are resent in the _same order_. This is because the messages will be hashed to be compared during follow up turns, to match requests to the correct session. If no hash matches everything will still work, but slower. Because a fresh context will be used and passed messages will be reingested.

##### Function Calling
Parallel function calls are working with node-llama-cpp + [functionary models](https://functionary.meetkai.com/). Other models, like llama3 instruct, also work with function calling (but no parallel calls). `tool_choice` can currently not be controlled and will always be `auto`.

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
- [x] Improve tests for longer conversations / context window shifting
- [x] Embeddings APIs
- [x] Improve node-llama-cpp token usage counts / TokenMeter
- [x] Reuse download logic from node-llama-cpp to support split ggufs.
- [x] Support preloading instances with context, like a long system message or few shot examples
- [ ] Document function call handler usage
- [ ] Nicer grammar api / loading
- [ ] Tests for request cancellation and timeouts
- [ ] Improve tests for embeddings
- [ ] A mock engine implementing for testing and examples / Allow user to customize engine implementations
- [ ] Logprobs support
- [ ] Improve offline support
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