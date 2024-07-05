### HTTP API

Note that the HTTP API is currently not secure (ie it's probably DoS-able, only minimal input validation). You should not host this on a public server without additional protections.

On the packaged web server there is currently only one additional HTTP endpoint:

- `GET /` - Prints info about spawned instances, available models and ongoing downloads.

#### OpenAI-Style API

`/openai/v1` is the default base path. The following endpoints and parameters are supported:

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

Some additional llama.cpp specific parameters are supported:

| Non-spec params     | gpt4all | node-llama-cpp |
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
| Grammar               | ❌      | ✅             |
| Function Calling      | ❌      | ✅             |

#### Usage

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
      // supported: 'text-completion', 'embedding', 'image-to-text', 'speech-to-text'
      task: 'text-completion',
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
      // Configure hardware / device to use.
      deviceOptions: {
        // GPU will be used automatically if available.
        // Only one model can use the gpu at a time.
        // gpu: true, // Supported for all
        // cpuThreads: 4, // Supported for gpt4all and node-llama-cpp
        // memLock: true, // Supported for node-llama-cpp.
      },
    },
  },
  // HTTP listen options. If you don't need a web server, use `startModelServer` or `new ModelServer()`.
  // Apart from `listen` they take the same configuration.
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