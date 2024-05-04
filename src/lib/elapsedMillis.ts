

export function elapsedMillis(start: bigint): number {
	const end = process.hrtime.bigint()
	return Number(end - BigInt(start)) / 1e6
}