
### On GPU usage

Only one model instance can run on gpu at a time. Instances can not switch between gpu and cpu. If left unconfigured, the first spawned instance of a model will automatically acquire gpu lock and use it. Note that if `minInstances` is set to something greater than 0 then the order in which models are configured will matter because initial instances will also be spawned in that order.

Automatic / unconfigured gpu usage:

```ts
{
  models: {
    'model1': {
      task: 'text-completion',
      engine: 'gpt4all',
      url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
      // first instance will automatically pick up gpu, then a second one will be spawned on cpu
      minInstances: 2,
    },
    'model2': {
      task: 'text-completion',
      engine: 'gpt4all',
      url: 'https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF/resolve/main/Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf',
      minInstances: 1, // this will always spawn on cpu because model1 already auto locked gpu
    },
  },
}
```

Another practical strategy is to explicitly configure gpu usage. The same model can be configured multiple times with different options so that the gpu instance can be targeted specifically. Like

```ts
{
  models: {
    'model1': {
      task: 'text-completion',
      engine: 'gpt4all',
      url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
      device: { gpu: true }, // this is effectively also maxInstances: 1
    },
    'model2': {
      task: 'text-completion',
      engine: 'gpt4all',
      url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
      // will spawn up to 2 cpu instances
      device: { gpu: false },
      maxInstances: 2,
    },
  },
}
```

It is possible to configure multiple models to use gpu, but only one gpu instance can be utilized at a time. If simultaneous requests to gpu models come in, each request will wait for the processing instance to release gpu lock before it can start. Then, depending on which models are requested in which order, instances may be disposed and spawned, or reused.

```ts
{
  models: {
    'model1': {
      task: 'text-completion',
      engine: 'gpt4all',
      url: 'https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf',
      device: { gpu: true },
    },
    'model2': {
      task: 'text-completion',
      engine: 'gpt4all',
      url: 'https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF/resolve/main/Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf',
      device: { gpu: true },
    },
  },
}
```

Note that switching the model that runs on gpu a lot (iE incoming requests `model1->model2->model1->model2) will lead to inefficient cache usage for chat completions and generally make requests slower, because models need to be unloaded and loaded + chat history has to be reingested.