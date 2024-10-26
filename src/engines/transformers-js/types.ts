import type {
	Processor,
	PreTrainedModel,
	PreTrainedTokenizer,
	PretrainedMixin,
	AutoProcessor,
	// DataType, // this is the tensor.js DataType Type, not the one im looking for
} from '@huggingface/transformers'
// import type { DataType } from '@huggingface/transformers/src/utils/dtypes.js' // this is dtypes.js DataType Type, cant import

export type TransformersJsModelClass = typeof PreTrainedModel
export type TransformersJsTokenizerClass = typeof PreTrainedTokenizer
export interface TransformersJsProcessorClass {
	from_pretrained: (typeof AutoProcessor)['from_pretrained']
}

export type TransformersJsDataType =
	| 'fp32'
	| 'fp16'
	| 'q8'
	| 'int8'
	| 'uint8'
	| 'q4'
	| 'bnb4'
	| 'q4f16'
