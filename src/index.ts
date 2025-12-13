import{
	env,
	pipeline,
	AutoTokenizer,
	AutoModelForSequenceClassification,
	TextGenerationPipeline,
	ProgressInfo,
	ProgressCallback,
	SummarizationPipeline,
	FeatureExtractionPipeline,
	PreTrainedTokenizer,
	TextGenerationOutput,
	TextGenerationSingle
}from '@huggingface/transformers';
import {TransformersEnvironment} from '@huggingface/transformers/types/env';
import {GenerationConfig} from '@huggingface/transformers/types/generation/configuration_utils';
import {MemoryVectorStore} from '@langchain/classic/vectorstores/memory';
import {Embeddings} from '@langchain/core/embeddings';
import lodash from 'lodash';

export class AICompareCandidates extends Embeddings{
	readonly env:TransformersEnvironment=env;
	DEBUG=true;

	generator:TextGenerationPipeline|null=null;
	generatorModelName='Xenova/LaMini-GPT-774m';
	generatorPromise:Promise<TextGenerationPipeline>|null=null;
	generatorProgressInfo:ProgressInfo=<ProgressInfo>{};
	generatorProgressCallback:ProgressCallback|null=null;

	summariser:SummarizationPipeline|null=null;
	summariserModelName='Xenova/distilbart-cnn-12-6';
	summariserPromise:Promise<SummarizationPipeline>|null=null;
	summariserProgressInfo:ProgressInfo=<ProgressInfo>{};
	summariserProgressCallback:ProgressCallback|null=null;

	embedder:FeatureExtractionPipeline|null=null;
	embedderModelName='Xenova/all-MiniLM-L6-v2';
	embedderPromise:Promise<FeatureExtractionPipeline>|null=null;
	embedderProgressInfo:ProgressInfo=<ProgressInfo>{};
	embedderProgressCallback:ProgressCallback|null=null;

	tokeniser:PreTrainedTokenizer|null;
	tokeniserModelName='Xenova/LaMini-GPT-774m';
	tokeniserPromise:Promise<PreTrainedTokenizer>|null=null;
	tokeniserProgressInfo:ProgressInfo=<ProgressInfo>{};
	tokeniserProgressCallback:ProgressCallback|null=null;

	generateSearchAreasMaxNewTokens=64;
	generateSearchAreasTemperature=0.35;
	generateSearchAreasRepetitionPenalty=1.5;

	rankingMaxNewTokens=64;
	rankingTemperature=0.35;
	rankingRepetitionPenalty=1.5;

	targetSummarisedStringTokenCount=420;

	static{
		env.allowRemoteModels=true;
		env.allowLocalModels=true;
	}
	
	async loadGenerator({
		progressCallback,
		modelName		
	}:{
		progressCallback?:ProgressCallback,
		modelName:string
	}={
		modelName:''
	}){
		if(typeof modelName==='string'&&modelName)this.generatorModelName=modelName;
		if(!this.generatorModelName)throw new Error('Invalid generator model name');
		if(progressCallback)this.generatorProgressCallback=progressCallback;
		//ts-ignore is needed due to frequent error TS2590: Expression produces a union type that is too complex to represent.
		//@ts-ignore
		this.generatorPromise=pipeline('text-generation',this.generatorModelName,{
			device:'webgpu',
			progress_callback:progressInfo=>{
				Object.assign(this.generatorProgressInfo,progressInfo);
				return this.generatorProgressCallback?.(progressInfo);
			}
		});
		this.generator=await this.generatorPromise;
		return this.generator;
	}

	async checkGeneratorLoaded(){
		if(!this.generatorPromise)this.loadGenerator();
		if(!this.generator)await this.generatorPromise;
		if(!this.generator)throw new Error('Unable to load generator');
	}

	async loadSummariser({
		progressCallback,
		modelName		
	}:{
		progressCallback?:ProgressCallback,
		modelName:string
	}={
		modelName:''
	}){
		if(typeof modelName==='string'&&modelName)this.summariserModelName=modelName;
		if(!this.summariserModelName)throw new Error('Invalid summariser model name');
		if(progressCallback)this.summariserProgressCallback=progressCallback;
		//ts-ignore is needed due to frequent error TS2590: Expression produces a union type that is too complex to represent.
		//@ts-ignore
		this.summariserPromise=pipeline('summarization',this.summariserModelName,{
			device:'webgpu',
			progress_callback:progressInfo=>{
				Object.assign(this.summariserProgressInfo,progressInfo);
				return this.summariserProgressCallback?.(progressInfo);
			}
		});
		this.summariser=await this.summariserPromise;
		return this.summariser;
	}

	async checkSummariserLoaded(){
		if(!this.summariserPromise)this.loadEmbedder();
		if(!this.summariser)await this.summariserPromise;
		if(!this.summariser)throw new Error('Unable to load summariser');
	}

	async loadEmbedder({
		progressCallback,
		modelName		
	}:{
		progressCallback?:ProgressCallback,
		modelName:string
	}={
		modelName:''
	}){
		if(typeof modelName==='string'&&modelName)this.embedderModelName=modelName;
		if(!this.embedderModelName)throw new Error('Invalid embedder model name');
		if(progressCallback)this.embedderProgressCallback=progressCallback;
		//ts-ignore is needed due to frequent error TS2590: Expression produces a union type that is too complex to represent.
		//@ts-ignore
		this.embedderPromise=pipeline('feature-extraction',this.embedderModelName,{
			device:'webgpu',
			progress_callback:progressInfo=>{
				Object.assign(this.embedderProgressInfo,progressInfo);
				return this.embedderProgressCallback?.(progressInfo);
			}
		});
		this.embedder=await this.embedderPromise;
		return this.embedder;
	}

	async checkEmbedderLoaded(){
		if(!this.embedderPromise)this.loadEmbedder();
		if(!this.embedder)await this.embedderPromise;
		if(!this.embedder)throw new Error('Unable to load embedder');
	}

	async loadTokeniser({
		progressCallback,
		modelName
	}:{
		progressCallback?:ProgressCallback,
		modelName:string
	}={
		modelName:''
	}){
		if(typeof modelName==='string'&&modelName)this.tokeniserModelName=modelName;
		if(!this.tokeniserModelName)throw new Error('Invalid tokeniser model name');
		if(progressCallback)this.tokeniserProgressCallback=progressCallback;
		//ts-ignore is needed due to frequent error TS2590: Expression produces a union type that is too complex to represent.
		//@ts-ignore
		this.tokeniserPromise=AutoTokenizer.from_pretrained(this.tokeniserModelName,{
			progress_callback:progressInfo=>{
				Object.assign(this.tokeniserProgressInfo,progressInfo);
				return this.tokeniserProgressCallback?.(progressInfo);
			}
		})
		this.tokeniser=await this.tokeniserPromise;
		return this.tokeniser;
	}

	async checkTokeniserLoaded(){
		if(!this.tokeniserPromise)this.loadTokeniser();
		if(!this.tokeniser)await this.tokeniserPromise;
		if(!this.tokeniser)throw new Error('Unable to load tokeniser');
	}

	async embedQuery(text:string):Promise<number[]>{
		await this.checkEmbedderLoaded();
		return Array.from((await this.embedder(text,{
			pooling:'mean',
			normalize:true
		})).data);
	}

	async embedDocuments(texts:string[]):Promise<number[][]>{
		return Promise.all(texts.map(text=>this.embedQuery(text)));
	}

	generatePromptTemplate(prompt:string){
		return 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'+
			'### Instruction:\n'+
			prompt+
			'\n\n### Response:';
	}

	async compareCandidates<Candidate>({
		candidates,
		problemDescription,
		generateSearchAreasInstruction,
		convertCandidateToDocument,
		candidatesForInitialSelection,
		candidatesForFinalSelection,
		generateRankingInstruction,
		extractIdentifiersFromRationale,
		candidateIdentifierField,
		getSummarisableSubstringIndices
	}:{
		candidates:Candidate[],
		problemDescription:string,
		generateSearchAreasInstruction:(problemDescription:string)=>string,
		convertCandidateToDocument:(candidate:Candidate)=>string,
		candidatesForInitialSelection:number,
		candidatesForFinalSelection:number,
		generateRankingInstruction:(problemDescription:string,summaries:string[])=>string,
		extractIdentifiersFromRationale:(rationale:string)=>string[],
		candidateIdentifierField:keyof Candidate,
		getSummarisableSubstringIndices?:(candidateDocument:string)=>{start:number,end:number}
	}):Promise<{
		selectedCandidates:Candidate[],
		rationale:string
	}>{
		let rationale='';
		let selectedCandidates:Candidate[]=[];

		await this.checkEmbedderLoaded();
		let candidateDocuments=candidates.map(candidate=>convertCandidateToDocument(candidate));
		let vectorStore=await MemoryVectorStore.fromTexts(
			lodash.cloneDeep(candidateDocuments),
			candidateDocuments.map((document,index)=>index),
			this
		);

		let searchAreasPromptTemplate=this.generatePromptTemplate(generateSearchAreasInstruction(problemDescription));
		if(this.DEBUG)console.log('Formatted search areas prompt: '+searchAreasPromptTemplate);
		await this.checkTokeniserLoaded();
		let searchAreasPromptTokens=this.tokeniser.encode(searchAreasPromptTemplate);
		if(searchAreasPromptTokens.length>this.tokeniser.model_max_length)throw new Error('Search areas instruction prompt is too long for the tokeniser model');

		await this.checkGeneratorLoaded();
		let pad_token_id=this.tokeniser.pad_token_id??this.tokeniser.sep_token_id??0;
		let eos_token_id=this.tokeniser.sep_token_id??2;
		let searchAreasReplyArray=await this.generator(searchAreasPromptTemplate,{
			max_new_tokens:this.generateSearchAreasMaxNewTokens,
			temperature:this.generateSearchAreasTemperature,
			repetition_penalty:this.generateSearchAreasRepetitionPenalty,
			pad_token_id,
			eos_token_id
		});
		let searchAreasReply=Array.isArray(searchAreasReplyArray?.[0])?searchAreasReplyArray?.[0]?.[0]:searchAreasReplyArray?.[0];
		if(!searchAreasReply.generated_text)throw new Error('No generated text for search areas');
		if(this.DEBUG)console.log('Generated search areas response: '+searchAreasReply.generated_text);
		let searchAreasResponseIndex=searchAreasReply.generated_text.toString().indexOf('### Response:');
		if(searchAreasResponseIndex>=0)searchAreasResponseIndex+='### Response:'.length;
		else searchAreasResponseIndex=0;

		let vectorSearchQuery=searchAreasReply.generated_text.toString().substring(searchAreasResponseIndex).trim();
		//generally the first sentence has the greatest relevance to the actual prompt
		if(vectorSearchQuery.includes('.'))vectorSearchQuery=vectorSearchQuery.split('.')[0].trim();
		if(this.DEBUG)console.log('Vector search query: '+vectorSearchQuery);
		let queryResult=await vectorStore.similaritySearch(vectorSearchQuery,candidatesForInitialSelection);

		await this.checkSummariserLoaded();
		let summaries=(await Promise.allSettled(queryResult.map(async result=>{
			if(!result.pageContent||typeof result.pageContent!=='string')return '';
			if(result.pageContent.trim().split(/\s+/).length<=this.targetSummarisedStringTokenCount)return result.pageContent;
			let summarisableSubstringIndices={
				start:0,
				end:result.pageContent.length
			};
			if(getSummarisableSubstringIndices)Object.assign(summarisableSubstringIndices,getSummarisableSubstringIndices(result.pageContent));
			summarisableSubstringIndices.start=lodash.clamp(lodash.toSafeInteger(summarisableSubstringIndices.start),0,result.pageContent.length);
			summarisableSubstringIndices.end=lodash.clamp(lodash.toSafeInteger(summarisableSubstringIndices.end),0,result.pageContent.length);
			let summarisableSubstring=result.pageContent.substring(summarisableSubstringIndices.start,summarisableSubstringIndices.end);
			let contentBefore=result.pageContent.substring(0,summarisableSubstringIndices.start);
			let contentAfter=result.pageContent.substring(summarisableSubstringIndices.end);
			let wordsWithoutSummarisable=contentBefore.split(/s+/).length+contentAfter.split(/s+/).length;
			let targetSummarisedSubstringTokenCount=Math.max(1,420-wordsWithoutSummarisable);
			let summarisedSubstringArray=await this.summariser(summarisableSubstring,<GenerationConfig>{
				max_length:targetSummarisedSubstringTokenCount
			});
			let summarisedSubstring=Array.isArray(summarisedSubstringArray?.[0])?summarisedSubstringArray?.[0]?.[0]:summarisedSubstringArray?.[0];
			let summarisedString=contentBefore+(summarisedSubstring?.summary_text??'').split(/s+/).slice(targetSummarisedSubstringTokenCount).join(' ')+contentAfter;
			if(this.DEBUG)console.log('Summarised candidate: '+summarisedString);
		}))).filter(result=>result.status==='fulfilled'&&result.value).map((result:PromiseFulfilledResult<string>)=>result.value);

		let rankingPromptTemplate=this.generatePromptTemplate(generateRankingInstruction(problemDescription,summaries));
		let rankingPromptTokens=this.tokeniser.encode(rankingPromptTemplate);
		if(rankingPromptTokens.length>this.tokeniser.model_max_length)throw new Error('Ranking instruction prompt is too long for the tokeniser model');
		let rankingArray=await this.generator(rankingPromptTemplate,{
			max_new_tokens:this.rankingMaxNewTokens,
			temperature:this.rankingTemperature,
			repetition_penalty:this.rankingRepetitionPenalty,
			pad_token_id,
			eos_token_id
		});
		let ranking=Array.isArray(rankingArray?.[0])?rankingArray?.[0]?.[0]:rankingArray[0];
		rationale=ranking.generated_text.toString().trim().replace(/(\*\*)|(<\/?s>)|(\[.*?\])\s*/g, '');
		let rationaleResponseIndex=rationale.indexOf('### Response:');
		if(rationaleResponseIndex>=0)rationaleResponseIndex+='### Response:'.length;
		else rationaleResponseIndex=0;
		rationale=rationale.substring(rationaleResponseIndex);
		if(!rationale)throw new Error('No rationale generated');
		let identifiers=extractIdentifiersFromRationale(rationale);
		if(identifiers.length>candidatesForFinalSelection)identifiers=identifiers.slice(0,candidatesForFinalSelection);
		selectedCandidates=identifiers.map(identifier=>{
			let selectedCandidate=candidates.find(candidate=>String(candidate[candidateIdentifierField]).toLowerCase()===identifier.toLowerCase());
			if(selectedCandidate)return selectedCandidate;
			selectedCandidate=candidates.find(candidate=>String(candidate[candidateIdentifierField]).toLowerCase().includes(identifier.toLowerCase()));
			if(selectedCandidate)return selectedCandidate;
			selectedCandidate=candidates.find(candidate=>identifier.toLowerCase().includes(String(candidate[candidateIdentifierField]).toLowerCase()));
			if(selectedCandidate)return selectedCandidate;
			return null;
		}).filter(Boolean);

		return{
			rationale,
			selectedCandidates
		};
	}
};

export default AICompareCandidates;