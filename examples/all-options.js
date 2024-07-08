import { startHTTPServer } from '../dist/http.js'

// Starts a http server for up to two instances of phi3 and serves them via openai API.
// startHTTPServer is only a thin wrapper around the ModelServer class that spawns a web server.
startHTTPServer({
	log: 'info', // 'debug', 'info', 'warn', 'error' - or pass a function as custom logger.
	// Limit how many instances may be handed out concurrently for processing.
	// If its exceeded, requests will stall until a model is available.
	// Defaults to 1 = process one request at a time.
	concurrency: 2,
	// Where to cache models to disk. Defaults to `~/.cache/lllms`
	// modelsPath: '/path/to/models',
	models: {
		// Specify as many models as you want. Identifiers can use a-zA-Z0-9_:\-\.
		// Required are `task`, `engine`, `url` and/or `file`.
		'phi3-mini-4k': {
			task: 'text-completion', // 'text-completion', 'embedding', 'image-to-text', 'speech-to-text'
			engine: 'node-llama-cpp', // 'node-llama-cpp', 'transformers-js', 'gpt4all'
			// Model weights may be specified by file and/or url.
			url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
			// specify sha256 hash to verify the downloaded file.
			sha256: '8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef',
			// File path may be absolute or relative to modelsPath.
			// - If an absolute path is given, it will be downloaded to that location.
			// - If a relative path is given, it will be resolved relative to modelsPath.
			file: 'Phi-3-mini-4k-instruct-q4.gguf',
			// The preparation process downloads and verifies models weights.
			// Use this to control when that happens. Defaults to 'on-demand'.
			// Options are:
			// - 'on-demand' = prepare on first request
			// - 'blocking' = prepare immediately on startup
			// - 'async' = prepare in background but don't block startup. Requests to the model during the preparation process will resolve once its ready.
			// Note that if minInstances > 0 then this is effectively always "blocking" because the model preparation will happen immediately.
			prepare: 'on-demand',
			// What should be preloaded in context, for text completion / chat models.
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
			// Options to control resource usage.
			contextSize: 2046, // Maximum context size. Will be determined automatically if not set.
			maxInstances: 2, // How many active sessions you wanna be able to cache at the same time.
			minInstances: 1, // To always keep at least one instance ready. Defaults to 0.
			ttl: 300, // Idle sessions will be disposed after this many seconds.
			// Set defaults for completions. These can be overridden per request.
			// If unset, default values depend on the engine.
			completionDefaults: {
				temperature: 1,
			},
			// Configure hardware / device to use.
			device: {
				// GPU will be used automatically if left unset.
				// Only one model can use the gpu at a time.
				// gpu: true, // Force gpu use for instance of this model. (This effectively limits maxInstance to 1.)
				// cpuThreads: 4, // Only gpt4all and node-llama-cpp
				// memLock: true, // Only node-llama-cpp.
			},
		},
	},
	// HTTP listen options. If you don't need a web server, use `startModelServer` or `new ModelServer()`.
	// Accepted arguments are identical, apart from `listen`.
	listen: {
		port: 3000,
	},
})
