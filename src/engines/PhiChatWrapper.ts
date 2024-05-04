import { ChatWrapper } from 'node-llama-cpp'
import {
	SpecialToken,
	LlamaText,
	SpecialTokensText,
	ChatHistoryItem,
	ChatModelFunctions,
} from 'node-llama-cpp'

export class PhiChatWrapper extends ChatWrapper {
	public readonly wrapperName: string = 'PhiChatWrapper'
	
	
	private stopGenerationTriggers: string[] | null = null
	public setStopGenerationTriggers(triggers: string[] | null) {
		this.stopGenerationTriggers = triggers
	}

	public override generateContextText(
		history: readonly ChatHistoryItem[],
		{
			availableFunctions,
			documentFunctionParams,
		}: {
			availableFunctions?: ChatModelFunctions
			documentFunctionParams?: boolean
		} = {},
	): {
		contextText: LlamaText
		stopGenerationTriggers: LlamaText[]
		ignoreStartText?: LlamaText[]
		functionCall?: {
			initiallyEngaged: boolean
			disengageInitiallyEngaged: LlamaText[]
		}
	} {
		const historyWithFunctions =
			this.addAvailableFunctionsSystemMessageToHistory(
				history,
				availableFunctions,
				{
					documentParams: documentFunctionParams,
				},
			)

		const resultItems: Array<{
			system: string
			user: string
			model: string
		}> = []

		let systemTexts: string[] = []
		let userTexts: string[] = []
		let modelTexts: string[] = []
		let currentAggregateFocus: 'system' | null = null

		function flush() {
			if (
				systemTexts.length > 0 ||
				userTexts.length > 0 ||
				modelTexts.length > 0
			)
				resultItems.push({
					system: systemTexts.join('\n\n'),
					user: userTexts.join('\n\n'),
					model: modelTexts.join('\n\n'),
				})

			systemTexts = []
			userTexts = []
			modelTexts = []
		}

		for (const item of historyWithFunctions) {
			if (item.type === 'system') {
				if (currentAggregateFocus !== 'system') flush()

				currentAggregateFocus = 'system'
				systemTexts.push(item.text)
			} else if (item.type === 'user') {
				flush()

				currentAggregateFocus = null
				userTexts.push(item.text)
			} else if (item.type === 'model') {
				flush()

				currentAggregateFocus = null
				modelTexts.push(this.generateModelResponseText(item.response))
			} else void (item satisfies never)
		}

		flush()

		const contextText = LlamaText(
			new SpecialToken('BOS'),
			resultItems.map(({ system, user, model }, index) => {
				const isLastItem = index === resultItems.length - 1

				return LlamaText([
					system.length === 0
						? LlamaText([])
						: LlamaText([
								new SpecialTokensText('<|system|>\n'),
								system,
								new SpecialTokensText('<|end|>\n'),
						  ]),

					user.length === 0
						? LlamaText([])
						: LlamaText([
								new SpecialTokensText('<|user|>\n'),
								user,
								new SpecialTokensText('<|end|>\n'),
						  ]),

					model.length === 0 && !isLastItem
						? LlamaText([])
						: LlamaText([
								new SpecialTokensText('<|assistant|>\n'),
								model,

								isLastItem
									? LlamaText([])
									: new SpecialTokensText('<|end|>\n'),
						  ]),
				])
			}),
		)
		
		const stopGenerationTriggers = [
			LlamaText(new SpecialToken('EOS')),
			LlamaText(new SpecialTokensText('<|end|>')),
			LlamaText('<|end|>'),
		]
		
		if (this.stopGenerationTriggers) {
			stopGenerationTriggers.push(
				...this.stopGenerationTriggers.map((trigger) => LlamaText(trigger)),
			)
		}

		return {
			contextText,
			stopGenerationTriggers,
		}
	}
}
