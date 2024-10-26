
### Engines

Currently the built-in inference engines are `node-llama-cpp`, `gpt4all` and `transformers-js` (highly experimental). Install the corresponding peer dependency before using an engine.

#### node-llama-cpp

Can be used for `text-completion` and `embedding` tasks. See the [node-llama-cpp docs](https://node-llama-cpp.withcat.ai/) for more information.

Find available GGUF models on [huggingface.co](https://huggingface.co/models?library=gguf).

#### gpt4all

Can be used for `text-completion` and `embedding` tasks. You can find parameter docs [here](https://github.com/nomic-ai/gpt4all/blob/c73f0e5c8c25ede56e3eeb28ff9dd37f09212994/gpt4all-bindings/typescript/src/gpt4all.d.ts#L615).

You can find available models [here](https://github.com/nomic-ai/gpt4all/blob/c73f0e5c8c25ede56e3eeb28ff9dd37f09212994/gpt4all-chat/metadata/models3.json)

#### transformers-js

Currently supporting `speech-to-text` and `image-to-text` tasks. See [tests](./tests/engines/transformers.test.ts).

#### node-stable-diffusion-cpp

WIP. See [tests](./tests/engines/stable-diffusion.test.ts).

#### Custom Engines

You can also write your own engine implementation. See [./src/engines](./src/engines) for how the built-in engines are implemented and [here](./tests/engines/experiments.test.ts) for examples of how to utilize custom engines to combine models and add multimodality to your chat completion endpoint. (Or to any other consumer of the ModelServer class.) Multiple ModelServers are allowed and can also be nested to create more complex pipelines.
