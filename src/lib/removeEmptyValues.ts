export function removeEmptyValues<T extends Record<string, any>>(dict: T): T {
	return Object.fromEntries(
		Object.entries(dict).filter(([_, v]) => {
			return v !== null && v !== undefined
		}),
	) as T
}
