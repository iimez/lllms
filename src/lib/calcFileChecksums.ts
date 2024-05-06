import fs from 'node:fs'
import crypto, { Hash } from 'node:crypto'

// stream file once to calculate multiple checksums
export function calcFileChecksums(
	filePath: string,
	hashTypes: string[] = ['md5', 'sha256'],
): Promise<Record<string, string>> {
	return new Promise((resolve, reject) => {
		const fileStream = fs.createReadStream(filePath)
		const hashes: Record<string, Hash> = {}
		for (const hashType of hashTypes) {
			hashes[hashType] = crypto.createHash(hashType)
		}
		fileStream.on('error', reject)
		fileStream.on('data', (chunk) => {
			for (const hash of Object.values(hashes)) {
				hash.update(chunk)
			}
		})
		fileStream.on('end', () => {
			resolve(
				Object.fromEntries(
					Object.entries(hashes).map(([hashType, hash]) => [
						hashType,
						hash.digest('hex'),
					]),
				),
			)
		})
	})
}
