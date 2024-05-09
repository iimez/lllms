import fs from 'node:fs'
import path from 'node:path'

export function loadGBNFGrammars(grammarsPath: string) {
	const gbnfFiles = fs
		.readdirSync(grammarsPath)
		.filter((f) => f.endsWith('.gbnf'))
	const grammars: Record<string, string> = {}
	for (const file of gbnfFiles) {
		const grammar = fs.readFileSync(path.join(grammarsPath, file), 'utf-8')
		grammars[file.replace('.gbnf', '')] = grammar
	}
	return grammars
}

