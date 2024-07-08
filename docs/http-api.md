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

Some additional llama.cpp- and gpt4all specific parameters are supported:

| Non-spec params     | gpt4all | node-llama-cpp |
| ------------------- | ------- | -------------- |
| top_k               | ✅      | ✅             |
| min_p               | ✅      | ✅             |
| repeat_penalty_num  | ✅      | ✅             |
| repeat_penalty      | ✅      | -              |

#### Functionality

| Feature               | gpt4all | node-llama-cpp |
| --------------------- | ------- | -------------- |
| Streaming             | ✅      | ✅             |
| Chat context cache    | ✅      | ✅             |
| System prompt         | ✅      | ✅             |
| Grammar               | ❌      | ✅             |
| Function Calling      | ❌      | ✅             |

#### Usage

```js lllms.js
import { startHTTPServer } from 'lllms'

// Starts a http server for up to two instances of phi3 and serves them via openai API.
// startHTTPServer is only a thin wrapper around the ModelServer class that spawns a web server.
startHTTPServer({
  concurrency: 2,
  models: {
    'phi3-mini-4k': {
      task: 'text-completion',
      engine: 'node-llama-cpp',
      url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
      preload: {
        // Note that for preloading to be utilized, requests must
        // also have these leading messages before the user message.
        messages: [
          {
            role: 'system',
            content: 'You are the Batman.',
          },
        ],
      },
      // Use these to control resource usage.
      contextSize: 1024, // Maximum context size. Will be determined automatically if not set.
      maxInstances: 2, // How many active sessions you wanna be able to cache at the same time.
      minInstances: 1, // To always keep at least one instance ready. Defaults to 0.
      ttl: 300, // Idle sessions will be disposed after this many seconds.
      // Set defaults for completions. These can be overridden per request.
      // If unset, default values depend on the engine.
      completionDefaults: {
        temperature: 1,
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
              "role": "system",
              "content": "You are the Batman."
          },
          {
              "role": "user",
              "content": "im robin, lets count to 10!"
          }
      ]
  }'
```
```json
{
  "id": "phi3-mini-4k:pfBGvlYg-z6dPZUn9",
  "model": "phi3-mini-4k",
  "object": "chat.completion",
  "created": 1720412918,
  "system_fingerprint": "b38af554bea1fb9867db54ebeff59d0590c5ce48",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello, Robin! As Batman, my focus is on protecting Gotham City and ensuring justice prevails. However, let's have a quick exercise to lighten the mood. Ready?\n\n1... 2... 3... 4... 5... 6... 7... 8... 9... And 10! Great job!\n\nRemember, my mission as Batman never ends, but it's always good to recharge and have fun alongside our partners. Let's keep Gotham safe together."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 118,
    "total_tokens": 130
  }
}
```