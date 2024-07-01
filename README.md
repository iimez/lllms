## lllms

Local Large Language Models for node.js. Simple tools to build complex AI applications on localhost. Built on [llama.cpp](https://github.com/ggerganov/llama.cpp/) via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) and [gpt4all](https://github.com/nomic-ai/gpt4all). And [transformers.js](https://github.com/xenova/transformers.js/)!

Prioritizing ease of use (DX and deployment) and nice APIs. For other approaches see [here](#related-solutions).

⚠️ This package is currently in beta. Some APIs may change. Feel free to report any issues you encounter.

### Features

- Configure as many models as you want, download and cache them on demand to `~/.cache/lllms`. Or provide them as abs file paths.
- Adjust the pools `concurrency`, and the models `maxInstances`, `ttl` and `contextSize` to fit your usecase. Can be tuned to either use no resources when idle or to always keep a model ready with context loaded.
- A chat session cache that will effectively reuse context across multiple turns or stateless requests.
- OpenAI spec API endpoints. See [#progress](#progress) for compatibility. 
- BYO web server or use the provided express server and middleware.
- Or don't use a web server and instead use the ModelServer directly within your node.js application.
- Have as many ModelServers running as you want, they can share the same cache directory.
- Use [custom engines](#engines) to combine multiple models (or do RAG) behind the scenes.

### Usage

#### JavaScript APIs

More details on the usage of `ModelServer` and the underlying `ModelPool` class can be found in [./examples/cli-chat](./examples/cli-chat.js) and [./examples/pool](./examples/pool.js).

To attach lllms to your existing (express, or any other) web server see [./examples/express-openai](./examples/express-openai.js) and [./examples/server-node](./examples/server-node.js). See [./src/http.ts](./src/http.ts) for more ways to integrate with existing HTTP servers.

For function calling support see the test [here](./tests/engines/lib/feature-functions.ts).

A one liner to start a server with a single model:
```js lllms.js
import { startHTTPServer } from 'lllms'

// Starts a http server for up to two instances of phi3 and serves them
// via openai API. It's only thin wrapper around the ModelServer class.
startHTTPServer({
  // Limit how many instances can be handed out concurrently, to handle
  // incoming requests. If its exceeded, requests will stall until a model
  // is available. Defaults to 1 = handle one request at a time.
  concurrency: 2,
  // Where to cache models to disk. Defaults to `~/.cache/lllms`
  // modelsPath: '/path/to/models',
  models: {
    // Specify as many models as you want. Identifiers can use a-zA-Z0-9_:\-\.
    // Required are `task`, `engine`, `url` and/or `file`.
    'phi3-mini-4k': {
      task: 'text-completion', // Use 'text-completion' or 'embedding'
      engine: 'node-llama-cpp', // 'node-llama-cpp', 'transformers-js', 'gpt4all'
      // Model weights may be specified by file and/or url.
      url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
      // File path may be absolute or relative to modelsPath.
      // If it does not exist and a url is configured it will be downloaded here.
      file: 'Phi-3-mini-4k-instruct-q4.gguf',
      // When to download and verify models weights.
      // Defaults to 'on-demand' = download on first request.
      // - 'blocking' = dont start before its downloaded
      // - 'async' = startup and download in background
      // Note that if minInstances > 0 is the same as "blocking"
      prepare: 'on-demand',
      // What should be preloaded in context, for text completion models.
      preload: {
        // Note that for preloading to be utilized, requests must
        // also have these leading messages before the user message.
        messages: [
          {
            role: 'system',
            content: 'You are a helpful assistant.',
          },
        ],
      },
      // Use these to control resource usage.
      contextSize: 4096, // Maximum context size. Will be determined automatically if not set.
      maxInstances: 2, // How many active sessions you wanna be able to cache at the same time.
      minInstances: 1, // To always keep at least one instance ready. Defaults to 0.
      ttl: 300, // Idle sessions will be disposed after this many seconds.
      // Set defaults for completions. These can be overridden per request.
      // If unset, default values depend on the engine.
      completionDefaults: {
        temperature: 1,
      },
      // Available options and their defaults are engine specific.
      engineOptions: {
        // GPU will be used automatically if available.
        // Only one model can use the gpu at a time.
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
// During download requests to a model will stall to get processed once the model is ready.
// http://localhost:3000 will serve a JSON of the current state of the server.
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

#### Engines

Currently the built-in inference engines are `node-llama-cpp`, `gpt4all` and `transformers-js` (highly experimental). You can also provide your own engine implementation. See [./src/engines](./src/engines) for how the built-in engines are implemented and [here](./tests/engines/custom-test.ts) for an example of how to utilize a custom engine to add support for vision / image content part messages to the OpenAI chat completion endpoint. (Or any other consumer of the ModelServer class.) Multiple ModelServers are allowed and can also be nested to create more complex pipelines.

#### HTTP API

Note that the API is currently not secure (ie the HTTP API is probably DoS-able, only minimal input validation). Misconfiguration or misuse can lead to process crashes or high memory usage. There are no additional safeguards in place and node.js memory limits will not apply. You should not host this on a public server without additional protections.

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
| Grammar               | ❌      | ✅             |
| Function Calling      | ❌      | ✅             |


#### Limitations and Known Issues

##### GPU Support limited to one model
Llama.cpp bindings currently do not support running multiple models on gpu at the same time. This means if `gpu` is set to `true` for a model, only one instance of it can run at one time (additional instances will refuse to spawn and requests will stall). If `gpu` is left unset more cpu instances can be spawned instead (and the first instance will still pick up gpu). Instances can not switch between gpu and cpu.

I hope I can improve this in the future. Meanwhile if you require closer control over which requests exactly use gpu, you can always configure a model multiple times with different configuration.

##### System Messages
System role messages are supported only as the first message in a chat completion session. All other system messages will be ignored.

##### Chat Context Cache
Note that the current context cache implementation only works if (apart from the final user message) the _same messages_ are resent in the _same order_. This is because the messages will be hashed to be compared during follow up turns, to match requests to the correct session. If no hash matches everything will still work, but slower. Because a fresh context will be used and passed messages will be reingested.

##### Function Calling
Only available when using node-llama-cpp and models that support it, like [functionary models](https://functionary.meetkai.com/) and Llama3 instruct. `tool_choice` can currently not be controlled and will always be `auto`. GBNF grammars cannot be used together with function calling.

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
- [x] transformers.js engine
- [x] Support custom engine implementations
- [x] Make sure nothing clashes if multiple servers/stores are using the same cache directory
- [x] See if we can install supported engines as peer deps
- [ ] Support more transformer.js tasks / pipelines
- [ ] Restructure docs, add function calling & grammar usage docs
- [ ] Rework GPU+device usage / lock (Support multiple models on gpu in cases where its possible)
- [ ] Allow configuring total RAM/VRAM usage (See https://github.com/ggerganov/llama.cpp/issues/4315 Check estimates?)
- [ ] Tests for cancellation and timeouts
- [ ] Logprobs support
- [ ] Improve offline support (allow running `Engine.prepareModel` ahead of time)
- [ ] Replace express with tinyhttp?
-
### Contributing

If you know how to fill in any of the above checkboxes or have additional ideas you'd like to make happen, feel free to open an issue or PR.

#### Possible Future Goals

- Create a separate HTTP API thats independent of the OpenAI spec and stateful.
- Add a clientside library (React hooks?) for use of above API.
- Provide a CLI. (Launch a server via `lllms serve config.json|js`? `lllms prepare` to download everything needed to disk?)
- Provide a Docker image. And maybe a Prometheus endpoint.

#### Currently not the Goals

- Another facade to LLM hoster HTTP API's. The strengths here are local/private/offline use.
- Worry too much about authentication or rate limiting or misuse. Host this with caution.
- Some kind of distributed or multi-node setup. Too soon.
- Other common tooling like vector stores, Chat GUIs, etc.

### Related Solutions

If you look at this package, you might also want to take a look at these other solutions:

- [ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) - Uses llama.cpp and provides a HTTP API. Also has experimental OpenAI API compatibility.
- [llama.cpp Server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#llamacpp-http-server) - The official llama.cpp HTTP API.
- [VLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) - A more production ready solution for hosting large language models.
- [LM Studio](https://lmstudio.ai/docs/local-server) - Also has a local server.
- [LocalAI](https://github.com/mudler/LocalAI) - Similar project in go.