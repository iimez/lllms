import { inspect } from 'node:util'

export function elapsedMillis(since: bigint): number {
	const now = process.hrtime.bigint()
	return Number(now - BigInt(since)) / 1e6
}

export function omitEmptyValues<T extends Record<string, any>>(dict: T): T {
	return Object.fromEntries(
		Object.entries(dict).filter(([_, v]) => {
			return v !== null && v !== undefined
		}),
	) as T
}

export function mergeAbortSignals(
	signals: Array<AbortSignal | undefined>,
): AbortSignal {
	const controller = new AbortController()
	const onAbort = () => {
		controller.abort()
	}
	for (const signal of signals) {
		if (signal) {
			signal.addEventListener('abort', onAbort)
		}
	}
	return controller.signal
}

export function printActiveHandles() {
	//@ts-ignore
	const handles = process._getActiveHandles();
	//@ts-ignore
	const requests = process._getActiveRequests();

	console.log('Active Handles:', inspect(handles, { depth: 1 }));
	console.log('Active Requests:', inspect(requests, { depth: 1 }));
}
