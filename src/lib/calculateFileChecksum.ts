import fs from 'node:fs'
import crypto from 'node:crypto'

export function calculateFileChecksum(filePath: string, hashType = 'sha256'): Promise<string> {
	return new Promise((resolve, reject) => {
		const fileStream = fs.createReadStream(filePath)
		const result = crypto.createHash(hashType)
		fileStream.on('error', reject)
		fileStream.on('data', (chunk) => {
				result.update(chunk)
		})
		fileStream.on('end', () => {
			resolve(
				result.digest('hex')
			)
		})
	})
}