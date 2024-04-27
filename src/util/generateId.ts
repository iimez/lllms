import { customAlphabet } from 'nanoid'

// Define a custom alphabet excluding '_' and '-'
const customAlphabetWithoutUnderscoreHyphen =
	'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

// Generate a nanoid using the custom alphabet
export const generateId = customAlphabet(
	customAlphabetWithoutUnderscoreHyphen,
	8,
)
