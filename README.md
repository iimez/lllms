## lllms

Local Large Language Models for node.js - Tools to build AI applications on localhost. Built on [llama.cpp](https://github.com/ggerganov/llama.cpp/) via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) and [gpt4all](https://github.com/nomic-ai/gpt4all). And [transformers.js](https://github.com/xenova/transformers.js/) using [ONNX](https://github.com/microsoft/onnxruntime/tree/main/js#onnxruntime-node)!

The project includes a model resource pool and an optional HTTP API server. Model file management is abstracted away completely. Useful for small-scale chatbots, local assistants, or any applications where private+offline is critical. For other not node-based projects, check out the [related solutions](#related-solutions) section.

⚠️ This package is currently in beta. Some APIs may change. Things may break. Issue reports are very welcome.

### Features

- Configure as many models as you want, download and cache them on demand to `~/.cache/lllms`. Or provide them as abs file paths.
- Adjust the pool `concurrency`, and the models `maxInstances`, `ttl` and `contextSize` to fit your usecase.
- Can be tuned to either use no resources when idle or to always keep a model ready with context preloaded.
- A chat session cache that will effectively reuse context across multiple turns or stateless requests.
- OpenAI spec API endpoints. See [HTTP API docs](./docs/http-api.md) for details. 
- BYO web server or use the provided express server and middleware.
- Or don't use a web server and instead use the JS APIs directly within your node.js application.
- Have as many ModelServers running as you want, they can share the same cache directory. (Multiple processes can as well)
- Use the [ModelPool](./examples/pool.js) class directly for a lowerlevel transaction-like API to manage model instances.
- Use [custom engines](./docs/engines.md#custom-engines) to combine multiple models (or do RAG) behind the scenes.

### Usage

Example with minimal configuration:

```ts basic.ts
import { ModelServer } from 'lllms'

const llms = new ModelServer({
  log: 'info', // default is 'warn'
  models: {
    'my-model': { // Identifiers can use a-zA-Z0-9_:\-\.
      // Required are `task`, `engine`, `url` and/or `file`.
      task: 'text-completion', // text-completion models can be used for chat and text generation tasks
      engine: 'node-llama-cpp', // don't forget to `npm install node-llama-cpp@beta`
      url: 'https://huggingface.co/HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/blob/main/smollm-135m-instruct-add-basics-q8_0.gguf',
    },
  },
})
await llms.start()
const result = await llms.processChatCompletionTask({
  model: 'my-model',
  messages: [
    { 
      role: 'user',
      content: 'Why are bananas rather blue than bread at night?',
    },
  ],
})
console.debug(result)
llms.stop()
```

Or, to start an OAI compatible HTTP server with two concurrent instances of the same model:

```ts http-api.ts
import { startHTTPServer } from 'lllms'
import OpenAI from 'openai'

const server = await startHTTPServer({
  listen: { port: 3000 }, // apart from `listen` options are identical to ModelServer
  concurrency: 2, // two inference processes may run at the same time
  models: {
    'smollm': {
      task: 'text-completion',
      engine: 'node-llama-cpp',
      url: 'https://huggingface.co/HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/blob/main/smollm-135m-instruct-add-basics-q8_0.gguf',
      maxInstances: 2, // two instances of this model may be loaded into memory
      device: {
        cpuThreads: 4, // limit cpu threads so we dont occupy all cores
      }
    },
  },
})

const client = new OpenAI({
  baseURL: 'http://localhost:3000/openai/v1/',
  apiKey: 'yes',
})
const completion = await client.beta.chat.completions.stream({
  stream_options: { include_usage: true },
  model: 'smollm',
  messages: [
    { role: 'user', content: 'lets count to 10, but only whisper every second number' },
  ],
})
for await (const chunk of completion) {
  if (chunk.choices[0]?.delta?.content) {
    process.stdout.write(chunk.choices[0].delta.content)
  }
}
server.stop()
```

More usage examples:
- Using all available options / model options API doc [./examples/all-options](./examples/all-options.js).
- Custom engines [./tests/engines/experiments.test.ts](./tests/engines/experiments.test.ts).
- A chat cli [./examples/chat-cli](./examples/chat-cli.js).
- `concurrency` behavior [./examples/concurrency](./examples/concurrency.js).
- Using the ModelPool directly [./examples/pool](./examples/pool.js).
- Using the express middleware [./examples/express](./examples/express.js).

Currently supported inference engines are:

| Engine | Peer Dependency |
| --- | --- |
| node-llama-cpp | `node-llama-cpp >= 3.0.0` |
| gpt4all | `gpt4all >= 4.0.0` |
| transformers-js | `@huggingface/transformers >= 3.0.0-alpha.9` |

See [engine docs](./docs/engines.md) for more information on each.

#### Limitations and Known Issues

##### Only one model can run on GPU at a time
Llama.cpp bindings currently do not support running multiple models on gpu at the same time. This can/will likely be improved in the future. See [GPU docs](./docs/gpu.md) for more information on how to work around that.

##### System Messages
System role messages are supported only as the first message in a chat completion session. All other system messages will be ignored. This is only for simplicity reasons and might change in the future.

##### Chat Context Cache
Note that the current context cache implementation only works if (apart from the final user message) the _same messages_ are resent in the _same order_. This is because the messages will be hashed to be compared during follow up turns, to match requests to the correct session. If no hash matches everything will still work, but slower. Because a fresh context will be used and the whole input conversation will be reingested, instead of just the new user message.

##### Function Calling
Only available when using node-llama-cpp and a model that supports function calling, like [functionary models](https://functionary.meetkai.com/) and Llama3 instruct. `tool_choice` can currently not be controlled and will always be `auto`. GBNF grammars cannot be used together with function calling.

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
- [x] Improve types, simpler node-llama-cpp grammar integration
- [x] Restructure docs, add function calling & grammar usage docs
- [x] TTL=0 should immediately dispose of instances instead of waiting (currently on avg 30s) for the next TTL check
- [x] Expose node-llama-cpp context shift strategy, lora, allow json schema as input for `grammar`
- [x] Improve types for tool definitions / json schema
- [x] Make pool dispose / stop more robust
- [x] Tests for cancellation and timeouts
- [x] transformer.js text embeddings
- [x] transformer.js image embeddings
- [x] transformer.js multimodal image/text embeddings (see [jina-clip-v1](https://github.com/xenova/transformers.js/issues/793) and [nomic-embed-vision](https://github.com/xenova/transformers.js/issues/848) issues.)
- [ ] utilize node-llama-cpp's support to reuse LlamaModel instances with multiple contexts
- [ ] Support transformer.js for text-completion tasks ([not yet supported in Node.js](https://github.com/xenova/transformers.js/blob/38a3bf6dab2265d9f0c2f613064535863194e6b9/src/models.js#L205-L207))
- [ ] Implement more transformer.js tasks (`imageToImage`, `textToImage`, `textToSpeech`?)
- [ ] non-chat text completions: Allow reuse of context
- [ ] non-chat text completions: Support preloading of prefixes
- [ ] Infill completion support https://github.com/withcatai/node-llama-cpp/blob/beta/src/evaluator/LlamaCompletion.ts#L322-L336
- [ ] Allow "prefilling" (partial) assistant responses like outlined [here](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response#how-to-prefill-claudes-response)
- [ ] Find a way to type available custom engines (and their options?)
- [ ] Rework GPU+device usage / lock (Support multiple models on gpu in cases where its possible)
- [ ] Add engine interfaces for resource use (and estimates, see https://github.com/ggerganov/llama.cpp/issues/4315 and https://github.com/withcatai/node-llama-cpp/blob/beta/src/gguf/insights/utils/resolveContextContextSizeOption.ts)
- [ ] Allow configuring a pools max memory usage
- [ ] Logprobs support
- [ ] Add transcript endpoint in oai api
- [ ] Add `n` parameter support to node-llama-cpp chat completions
- [ ] [CLI](https://github.com/iimez/lllms/discussions/7)
- [ ] Replace express with tinyhttp

### Contributing

If you know how to fill in any of the above checkboxes or have additional ideas you'd like to make happen, feel free to open an issue, PR or open a new discussion.

#### Possible Future Goals

- Create a separate HTTP API thats independent of the OpenAI spec and stateful. See [discussion](https://github.com/iimez/lllms/discussions/8).
- Add a clientside library (React hooks?) for use of above API.
- Provide a Docker image. And maybe a Prometheus endpoint.

#### Currently not the Goals

- A facade to LLM cloud hoster HTTP API's. The strengths here are local/private/offline use.
- Worry too much about authentication or rate limiting or misuse. Host this with caution.
- Some kind of distributed or multi-node setup. That should probably be something designed for this purpose from the ground up.
- Other common related tooling like vector stores, Chat GUIs, etc. Scope would probably get out of hand.

### Related Solutions

If you look at this package, you might also want to take a look at these other solutions:

- [ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) - Uses llama.cpp and provides a HTTP API. Also has experimental OpenAI API compatibility.
- [llama.cpp Server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#llamacpp-http-server) - The official llama.cpp HTTP API.
- [VLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) - A more production ready solution for hosting large language models.
- [LM Studio](https://lmstudio.ai/docs/local-server) - Also has a local server.
- [LocalAI](https://github.com/mudler/LocalAI) - Similar project in go.
- [Petals](https://github.com/bigscience-workshop/petals) - Local (and distributed!) inference in python.