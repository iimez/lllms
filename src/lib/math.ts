/**
 * Calculates the dot product of two vectors.
 * @param vecA - The first vector.
 * @param vecB - The second vector.
 * @returns The dot product of vecA and vecB.
 */
function dotProduct(vecA: number[], vecB: number[]): number {
	return vecA.reduce((sum, value, index) => sum + value * vecB[index], 0)
}

/**
 * Calculates the magnitude of a vector.
 * @param vec - The vector.
 * @returns The magnitude of the vector.
 */
function magnitude(vec: number[]): number {
	return Math.sqrt(vec.reduce((sum, value) => sum + value * value, 0))
}

/**
 * Calculates the cosine similarity between two vectors.
 * @param vecA - The first vector.
 * @param vecB - The second vector.
 * @returns The cosine similarity between vecA and vecB.
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
	const dotProd = dotProduct(vecA, vecB)
	const magnitudeA = magnitude(vecA)
	const magnitudeB = magnitude(vecB)
	return dotProd / (magnitudeA * magnitudeB)
}
