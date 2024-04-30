import { IncomingMessage, ServerResponse } from 'node:http'
import { existsSync, statSync } from 'node:fs'
import path from 'node:path'
import type { OpenAI } from 'openai'
import { LLMPool } from '../../../pool.js'

// https://platform.openai.com/docs/api-reference/models/list
export function createListModelsHandler(pool: LLMPool) {
	return async (req: IncomingMessage, res: ServerResponse) => {
		const data: OpenAI.Model[] = Object.entries(pool.config.models).map(
			([id, config]) => {
				let created = 0
				if (existsSync(config.file)) {
					//https://stackoverflow.com/a/51442878
					created = Math.floor(statSync(config.file).birthtime.getTime() / 1000)
				}
				// TODO possible to get created/owned_by from gguf metadata?
				return {
					object: 'model',
					owned_by: path.basename(config.file),
					id,
					created,
				}
			},
		)

		res.writeHead(200, { 'Content-Type': 'application/json' })
		res.end(JSON.stringify({ object: 'list', data }, null, 2))
	}
}
