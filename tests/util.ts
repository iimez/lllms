import { LLMServer } from '#lllms/server.js'
import { ChatCompletionRequest, CompletionRequest } from '#lllms/types/index.js'

const testDefaults = {
	model: 'test',
	temperature: 0,
	maxTokens: 64,
}
const defaultTimeout = 30000

export async function createChatCompletion(
	server: LLMServer,
	args: Omit<ChatCompletionRequest, 'model'> & { model?: string },
	timeout = defaultTimeout
) {
	const mergedArgs = {
		...testDefaults,
		...args,
	}
	const lock = await server.pool.requestInstance(mergedArgs)
	const handle = lock.instance.createChatCompletion(mergedArgs)
	const result = await handle.process({ timeout })
	const device = lock.instance.gpu ? 'gpu' : 'cpu'
	await lock.release()
	return { handle, result, device }
}

export async function createCompletion(
	server: LLMServer,
	args: Omit<CompletionRequest, 'model'> & { model?: string },
	timeout = defaultTimeout
) {
	const mergedArgs = {
		...testDefaults,
		...args,
	}
	const lock = await server.pool.requestInstance(mergedArgs)
	const handle = lock.instance.createCompletion(mergedArgs)
	const result = await handle.process({ timeout })
	const device = lock.instance.gpu ? 'gpu' : 'cpu'
	await lock.release()
	return { handle, result, device }
}

export function parseInstanceId(completionId: string) {
	// part after the last ":" because model names may include colons
	const afterModelName = completionId.split(':').pop()
	const instanceId = afterModelName?.split('-')[0] // rest is instanceId-completionId
	return instanceId
}