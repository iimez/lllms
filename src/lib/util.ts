
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

export function formatBytes(bytes: number, decimals = 2): string {
	if (bytes === 0) return '0 Bytes'
	const k = 1024
	const dm = decimals < 0 ? 0 : decimals
	const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
	const i = Math.floor(Math.log(bytes) / Math.log(k))
	return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

export function mergeAbortSignals(signals: AbortSignal[]): AbortSignal {
	const controller = new AbortController()
	const onAbort = () => {
		controller.abort()
	}
	for (const signal of signals) {
		signal.addEventListener('abort', onAbort)
	}
	return controller.signal
}