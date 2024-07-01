import {
	ChatMessage,
	MessageContentPart,
	MessageTextContentPart,
} from '#lllms/types/index.js'

function isTextContentPart(
	part: MessageContentPart,
): part is MessageTextContentPart {
	return part.type === 'text'
}

export function flattenMessageTextContent(content: ChatMessage['content']): string {
	if (typeof content === 'string') {
		return content
	}
	const parts = content.filter(isTextContentPart)
	return parts.map((part) => part.text).join('\n')
}
