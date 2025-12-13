import{
	env,
	pipeline,
	AutoTokenizer,
	AutoModelForSequenceClassification,
	TextGenerationPipeline,
	ProgressInfo,
	ProgressCallback
}from '@huggingface/transformers';

export class AICompareCandidates{
	generatorModelName='Xenova/LaMini-GPT-774m';
	generatorPromise:Promise<TextGenerationPipeline>|null=null;
	generatorProgressInfo:ProgressInfo|null=null;
	generatorProgressCallback:ProgressCallback|null=null;
	
	async loadGenerator(generatorProgressCallback?:ProgressCallback,generatorModelName=''){
		if(typeof generatorModelName==='string'&&generatorModelName)this.generatorModelName=generatorModelName;
		if(generatorProgressCallback)this.generatorProgressCallback=generatorProgressCallback;
		//ts-ignore is needed due to frequent error TS2590: Expression produces a union type that is too complex to represent.
		//@ts-ignore
		this.generatorPromise=pipeline('text-generation',this.generatorModelName,{
			device:'webgpu',
			progress_callback:progressInfo=>{
				this.generatorProgressInfo=progressInfo;
				return this.generatorProgressCallback(progressInfo);
			}
		});
	}
};

export default AICompareCandidates;