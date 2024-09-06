import fs from 'node:fs'

const activeLocks = new Set<string>()
const postfix = '.lock'

function clearFileLocks() {
	for (const lock of activeLocks) {
		clearFileLock(lock)
	}
}

process.on('SIGINT', () => {
	clearFileLocks()
	process.exit(0)
})

process.on('SIGTERM', () => {
	clearFileLocks()
})

process.on('exit', () => {
	clearFileLocks()
})

function clearFileLock(file: string) {
	activeLocks.delete(file)
	if (fs.existsSync(file + postfix)) {
		fs.rmSync(file + postfix)
	}
}

export function acquireFileLock(file: string, signal?: AbortSignal): Promise<() => void> {
	const lockFile = file + postfix
	
	return new Promise((resolve, reject) => {
		signal?.addEventListener('abort', () => {
			clearFileLock(file)
		})
		const interval = setInterval(() => {
			if (!fs.existsSync(lockFile)) {
				fs.writeFileSync(lockFile, '')
				activeLocks.add(file)
				clearInterval(interval)
				resolve(() => clearFileLock(file))
			}
		}, 1000)
	})
}
