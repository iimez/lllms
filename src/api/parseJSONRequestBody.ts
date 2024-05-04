import { IncomingMessage } from 'node:http'

export function parseJSONRequestBody(req: IncomingMessage | Request): Promise<any> {
	return new Promise((resolve, reject) => {

		// if request is from express theres no need to parse anything
		if ('body' in req) {
			resolve(req.body)
			return
		}
		
		// for native http server
		let body = ''
		req.on('data', (chunk) => {
			body += chunk.toString()
		})
		req.on('end', () => {
			try {
				const data = JSON.parse(body)
				resolve(data)
			} catch (error) {
				reject(error)
			}
		})
	})
}