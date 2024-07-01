import chalk from 'chalk'

export const LogLevels = {
	error: 'error',
	warn: 'warn',
	info: 'info',
	debug: 'debug',
	verbose: 'verbose',
} as const

export type LogLevel = keyof typeof LogLevels
export type Logger = (level: LogLevel, message: string, meta?: any) => void

export function withLogMeta(logger: Logger, meta: object) {
	return (level: LogLevel, message: string, extraMeta: object = {}) => {
		logger(level, message, { ...meta, ...extraMeta })
	}
}

export function createSublogger(
	minLevelOrLogger: LogLevel | Logger = LogLevels.warn,
) {
	if (minLevelOrLogger) {
		return typeof minLevelOrLogger === 'string'
			? createLogger(minLevelOrLogger)
			: minLevelOrLogger
	} else {
		return createLogger(LogLevels.warn)
	}
}

export function createLogger(minLevel: LogLevel) {
	const levels = Object.keys(LogLevels).reverse()
	const minLevelIndex = levels.indexOf(minLevel)

	return function log(level: LogLevel, message: string, meta?: any) {
		const levelIndex = levels.indexOf(level)
		if (levelIndex >= minLevelIndex) {
			const formattedMessage = formatMessage(level, message, meta)
			switch (level) {
				case LogLevels.error:
					console.error(formattedMessage, meta?.error)
					break
				case LogLevels.warn:
					console.warn(formattedMessage)
					break
				case LogLevels.info:
					console.info(formattedMessage)
					break
				case LogLevels.debug:
				case LogLevels.verbose:
					console.debug(formattedMessage)
					break
			}
		}
	}
}

function formatMessage(level: LogLevel, message: string, meta: any = {}) {
	const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 22)
	let messageStr = `[${timestamp}]`

	switch (level) {
		case LogLevels.error:
			messageStr += chalk.red(`[erro]`)
			break
		case LogLevels.warn:
			messageStr += chalk.yellow(`[warn]`)
			break
		case LogLevels.info:
			messageStr += chalk.white(`[info]`)
			break
		case LogLevels.debug:
			messageStr += chalk.gray(`[debg]`)
			break
		case LogLevels.verbose:
			messageStr += chalk.gray(`[verb]`)
			break
	}

	if (meta?.sequence) {
		messageStr += ' ' + chalk.cyan(meta.sequence)
	}

	if (meta?.instance) {
		messageStr += ' ' + chalk.magenta(meta.instance)
	} else if (meta?.model) {
		messageStr += ' ' + chalk.magenta(meta.model)
	}

	if (meta?.task) {
		messageStr += ' ' + chalk.green(meta.task)
	}

	messageStr += ' ' + message

	if (meta) {
		const { instance, sequence, model, task, elapsed, error, ...otherData } =
			meta
		if (elapsed) {
			if (elapsed < 1000) {
				messageStr += ' ' + chalk.magenta(`+${elapsed.toFixed(2)}ms`)
			} else {
				messageStr += ' ' + chalk.magenta(`+${(elapsed / 1000).toFixed(2)}s`)
			}
		}
		if (Object.keys(otherData).length > 0) {
			messageStr += ' ' + JSON.stringify(otherData, null, 2)
		}
	}
	return messageStr
}
