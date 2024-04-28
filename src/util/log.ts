import chalk from 'chalk'

export const LogLevels ={
	error: "error",
	warn: "warn",
	info: "info",
	debug: "debug",
	verbose: "verbose"
} as const

export type LogLevel = keyof typeof LogLevels
export type Logger = (level: LogLevel, message: string, meta?: any) => void

export function createLogger(minLevel: LogLevel) {
	const levels = Object.keys(LogLevels).reverse()
	const minLevelIndex = levels.indexOf(minLevel)
	
	return function (level: LogLevel, message: string, meta?: any) {
		const levelIndex = levels.indexOf(level)
		if (levelIndex >= minLevelIndex) {
			const formattedMessage = formatMessage(level, message, meta)
			switch (level) {
				case LogLevels.error:
					console.error(formattedMessage)
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

export function formatMessage(level: LogLevel, message: string, meta?: any) {
	const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 22)
	let messageStr = `[${timestamp}] `
	
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
	
	const { instance, elapsed, ...otherData } = meta
	if (instance) {
		messageStr += ` ${instance}`
		
	}

	messageStr += ' ' + message
	if (elapsed) {
		messageStr += ' ' + chalk.magenta(`in ${elapsed}ms`)
	}
	if (Object.keys(otherData).length > 0) {
		messageStr += ' ' + JSON.stringify(otherData, null, 2).replace(/\n/g, ' ').replace(/\s+/g, ' ')
	}
	return messageStr
}
