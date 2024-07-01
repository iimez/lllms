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
	// @ts-ignore
	const activeHandles = process._getActiveHandles()
	console.log('Active Handles:', activeHandles.length)
	activeHandles.forEach((handle: any, index: number) => {
		console.log(`Handle ${index + 1}:`, handle.constructor.name)
	})

	// @ts-ignore
	const activeRequests = process._getActiveRequests()
	console.log('Active Requests:', activeRequests.length)
	activeRequests.forEach((request: any, index: number) => {
		console.log(`Request ${index + 1}:`, request.constructor.name)
	})
}
