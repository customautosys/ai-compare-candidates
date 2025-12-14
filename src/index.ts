import{
	env,
	pipeline,
	AutoTokenizer,
	TextGenerationPipeline,
	ProgressInfo,
	ProgressCallback,
	SummarizationPipeline,
	FeatureExtractionPipeline,
	PreTrainedTokenizer,
	TextGenerationConfig
}from '@huggingface/transformers';
import {MemoryVectorStore} from '@langchain/classic/vectorstores/memory';
import {Embeddings} from '@langchain/core/embeddings';
import lodash from 'lodash';
import jsan from 'jsan';

export class AICompareCandidates extends Embeddings{
	readonly env=env;
	DEBUG=true;

	generator:TextGenerationPipeline|null=null;
	generatorModelName='Xenova/LaMini-GPT-124M';
	generatorPromise:Promise<TextGenerationPipeline>|null=null;
	generatorProgressInfo:ProgressInfo=<ProgressInfo>{};
	generatorProgressCallback:ProgressCallback|null=null;

	summariser:SummarizationPipeline|null=null;
	summariserModelName='Xenova/distilbart-cnn-12-6';
	summariserPromise:Promise<SummarizationPipeline>|null=null;
	summariserProgressInfo:ProgressInfo=<ProgressInfo>{};
	summariserProgressCallback:ProgressCallback|null=null;

	embedder:FeatureExtractionPipeline|null=null;
	embedderModelName='Xenova/all-MiniLM-L12-v2';
	embedderPromise:Promise<FeatureExtractionPipeline>|null=null;
	embedderProgressInfo:ProgressInfo=<ProgressInfo>{};
	embedderProgressCallback:ProgressCallback|null=null;

	tokeniser:PreTrainedTokenizer|null=null;
	tokeniserModelName=this.generatorModelName;
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
		env.localModelPath='';
		env.allowRemoteModels=true;
		env.allowLocalModels=false;
	}

	constructor(){
		super({});
	}
	
	async loadGenerator({
		progressCallback,
		modelName=''
	}:AICompareCandidates.LoadArguments=<AICompareCandidates.LoadArguments>{}){
		if(typeof modelName==='string'&&modelName)this.generatorModelName=modelName;
		if(!this.generatorModelName)throw new Error('Invalid generator model name');
		if(progressCallback)this.generatorProgressCallback=progressCallback;
		//ts-ignore is needed due to frequent error TS2590: Expression produces a union type that is too complex to represent.
		//@ts-ignore
		this.generatorPromise=pipeline('text-generation',this.generatorModelName,{
			device:'webgpu',
			progress_callback:progressInfo=>{
				if(this.DEBUG)console.log(jsan.stringify(progressInfo));
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
		modelName=''
	}:AICompareCandidates.LoadArguments=<AICompareCandidates.LoadArguments>{}){
		if(typeof modelName==='string'&&modelName)this.summariserModelName=modelName;
		if(!this.summariserModelName)throw new Error('Invalid summariser model name');
		if(progressCallback)this.summariserProgressCallback=progressCallback;
		//ts-ignore is needed due to frequent error TS2590: Expression produces a union type that is too complex to represent.
		//@ts-ignore
		this.summariserPromise=pipeline('summarization',this.summariserModelName,{
			device:'webgpu',
			progress_callback:progressInfo=>{
				if(this.DEBUG)console.log(jsan.stringify(progressInfo));
				Object.assign(this.summariserProgressInfo,progressInfo);
				return this.summariserProgressCallback?.(progressInfo);
			}
		});
		this.summariser=await this.summariserPromise;
		return this.summariser;
	}

	async checkSummariserLoaded(){
		if(!this.summariserPromise)this.loadSummariser();
		if(!this.summariser)await this.summariserPromise;
		if(!this.summariser)throw new Error('Unable to load summariser');
	}

	async loadEmbedder({
		progressCallback,
		modelName=''
	}:AICompareCandidates.LoadArguments=<AICompareCandidates.LoadArguments>{}){
		if(typeof modelName==='string'&&modelName)this.embedderModelName=modelName;
		if(!this.embedderModelName)throw new Error('Invalid embedder model name');
		if(progressCallback)this.embedderProgressCallback=progressCallback;
		//ts-ignore is needed due to frequent error TS2590: Expression produces a union type that is too complex to represent.
		//@ts-ignore
		this.embedderPromise=pipeline('feature-extraction',this.embedderModelName,{
			device:'webgpu',
			progress_callback:progressInfo=>{
				if(this.DEBUG)console.log(jsan.stringify(progressInfo));
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
		modelName=''
	}:AICompareCandidates.LoadArguments=<AICompareCandidates.LoadArguments>{}){
		if(typeof modelName==='string'&&modelName)this.tokeniserModelName=modelName;
		if(!this.tokeniserModelName)throw new Error('Invalid tokeniser model name');
		if(progressCallback)this.tokeniserProgressCallback=progressCallback;
		//ts-ignore is needed due to frequent error TS2590: Expression produces a union type that is too complex to represent.
		//@ts-ignore
		this.tokeniserPromise=AutoTokenizer.from_pretrained(this.tokeniserModelName,{
			progress_callback:progressInfo=>{
				if(this.DEBUG)console.log(jsan.stringify(progressInfo));
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
		return Array.from((await this.embedder?.(text,{
			pooling:'mean',
			normalize:true
		}))?.data);
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

	defaultGenerateSearchAreasInstruction(problemDescription:string){
		return 'List the relevant subject areas for the following issues. Limit your response to 100 words.\nIssues: "'+problemDescription+'"';
	}

	defaultConvertCandidateToDocument<Candidate>({
		candidate,
		index
	}:AICompareCandidates.ConvertCandidateToDocumentArguments<Candidate>=<AICompareCandidates.ConvertCandidateToDocumentArguments<Candidate>>{}){
		let document='Start of Candidate #'+index;
		for(let i in candidate)document+='\n'+lodash.startCase(i)+': '+(typeof candidate[i]==='object'?jsan.stringify(candidate[i]):String(candidate[i]));
		document+='\nEnd of Candidate #'+index;
		return document;
	}

	defaultGenerateRankingInstruction({
		problemDescription,
		summaries,
		candidatesForFinalSelection,
		candidateIdentifierField
	}:AICompareCandidates.GenerateRankingInstructionArguments=<AICompareCandidates.GenerateRankingInstructionArguments>{}){
		return 'Strictly follow these rules:\n'+
			'1. Rank ONLY the top '+candidatesForFinalSelection+' candidates with one 15-word sentence explaining why\n'+
			'2. Rank the candidates based on "'+problemDescription.replace(/(\r|\n)/g,' ')+'"\n'+
			'3. If unclear, say "Insufficient information to determine"\n\n'+
			'Candidates:\n\n'+summaries.join('\n\n')+'\n\n'+
			'Format exactly:\n'+
			'#1. "[Full '+lodash.startCase(candidateIdentifierField)+']": [15-word explanation]\n'+
			'#2. ...'
	}

	regexIndexOf(text:string,regex:RegExp,startIndex:number){
    	let indexInSuffix=text.slice(startIndex).search(regex);
    	return indexInSuffix<0?indexInSuffix:indexInSuffix+startIndex;
	}

	defaultExtractIdentifierFromCandidateDocument({
		candidateDocument,
		candidateIdentifierField
	}:AICompareCandidates.ExtractIdentifierFromCandidateDocumentArguments=<AICompareCandidates.ExtractIdentifierFromCandidateDocumentArguments>{}){
		let startCase=lodash.startCase(candidateIdentifierField);
		let startIndex=candidateDocument.indexOf(startCase);
		if(startIndex<0)startIndex=candidateDocument.toLowerCase().indexOf(startCase.toLowerCase());
		if(startIndex>=0)startIndex+=startCase.length;
		if(startIndex<0)startIndex=candidateDocument.toLowerCase().indexOf(candidateIdentifierField.toLowerCase());
		if(startIndex>=0)startIndex+=candidateIdentifierField.length;
		else return '';
		startIndex=candidateDocument.indexOf(':',startIndex);
		if(startIndex<0)startIndex=this.regexIndexOf(candidateDocument,/\s+/,startIndex);
		if(startIndex<0)return '';
		let endIndex=candidateDocument.indexOf('\n',startIndex);
		if(endIndex<0)endIndex=candidateDocument.length;
		return candidateDocument.substring(startIndex,endIndex).trim();
	}

	defaultExtractIdentifiersFromRationale(rationale:string){
		let regex=/^\s*#\s*\d+\s*\.?\s*"([^"]+)"/gm;
		let matches:string[]=[];
		for(let match:RegExpExecArray|null;Array.isArray(match=regex.exec(rationale));)if(match[1])matches.push(match[1]);
		return matches;
	}

	async compareCandidates<Candidate>({
		candidates,
		problemDescription='',
		generateSearchAreasInstruction=this.defaultGenerateSearchAreasInstruction.bind(this),
		convertCandidateToDocument=this.defaultConvertCandidateToDocument.bind(this),
		candidatesForInitialSelection=2,
		candidatesForFinalSelection=1,
		generateRankingInstruction=this.defaultGenerateRankingInstruction.bind(this),
		extractIdentifiersFromRationale=this.defaultExtractIdentifiersFromRationale.bind(this),
		extractIdentifierFromCandidateDocument=this.defaultExtractIdentifierFromCandidateDocument.bind(this),
		candidateIdentifierField=undefined,
		getSummarisableSubstringIndices
	}:AICompareCandidates.CompareArguments<Candidate>=<AICompareCandidates.CompareArguments<Candidate>>{}):Promise<AICompareCandidates.CompareCandidatesReturn<Candidate>|void>{
		if(!Array.isArray(candidates)||candidates.length<=0)throw new Error('No candidates provided');
		candidatesForInitialSelection=lodash.toSafeInteger(candidatesForInitialSelection);
		if(candidatesForInitialSelection<=0)throw new Error('Candidates for initial selection must be a positive integer bigger than 0');
		candidatesForFinalSelection=lodash.toSafeInteger(candidatesForFinalSelection);
		if(candidatesForFinalSelection<=0)throw new Error('Candidates for initial selection must be a positive integer bigger than 0');
		if(candidatesForInitialSelection<candidatesForFinalSelection)throw new Error('Candidates for initial selection must be equal or more than candidates for final selection');
		if(candidatesForInitialSelection>candidates.length)throw new Error('There are '+candidatesForInitialSelection+'candidates for initial selection which is more than the total number of candidates of '+candidates.length);
		if(candidatesForFinalSelection>candidates.length)throw new Error('There are '+candidatesForFinalSelection+'candidates for initial selection which is more than the total number of candidates of '+candidates.length);
		if(!candidateIdentifierField){
			candidateIdentifierField=Object.keys(candidates[0] as object)[0] as keyof Candidate;
			if(!candidateIdentifierField)throw new Error('No candidate identifier field');
		}

		let rationale='';
		let selectedCandidates:Candidate[]=[];

		await this.checkEmbedderLoaded();
		if(!this.embedder)return;
		let candidateDocuments=candidates.map((candidate,index)=>convertCandidateToDocument({
			candidate,
			index
		}));
		let vectorStore=await MemoryVectorStore.fromTexts(
			lodash.cloneDeep(candidateDocuments),
			candidateDocuments.map((document,index)=>index),
			this
		);

		let searchAreasPromptTemplate=this.generatePromptTemplate(generateSearchAreasInstruction(problemDescription));
		if(this.DEBUG)console.log('Formatted search areas prompt: '+searchAreasPromptTemplate);
		await this.checkTokeniserLoaded();
		if(!this.tokeniser)return;
		let searchAreasPromptTokens=this.tokeniser.encode(searchAreasPromptTemplate);
		if(searchAreasPromptTokens.length>this.tokeniser.model_max_length)throw new Error('Search areas instruction prompt is too long for the tokeniser model');

		await this.checkGeneratorLoaded();
		if(!this.generator)return;
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

		let summaries:string[]=[];
		//only bother doing summarisation if there are candidates which exceed the token count
		if(queryResult.some(result=>result.pageContent.trim().split(/\s+/).length>this.targetSummarisedStringTokenCount)){
			await this.checkSummariserLoaded();
			if(!this.summariser)return;
			summaries=(await Promise.allSettled(queryResult.map(async result=>{
				if(!result.pageContent||typeof result.pageContent!=='string')return '';
				if(result.pageContent.trim().split(/\s+/).length<=this.targetSummarisedStringTokenCount)return result.pageContent;
				let summarisableSubstringIndices:AICompareCandidates.SummarisableSubstringIndices={
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
				let summarisedSubstringArray=await this.summariser?.(summarisableSubstring,<TextGenerationConfig>{
					max_length:targetSummarisedSubstringTokenCount
				});
				let summarisedSubstring=Array.isArray(summarisedSubstringArray?.[0])?summarisedSubstringArray?.[0]?.[0]:summarisedSubstringArray?.[0];
				let summarisedString=contentBefore+(summarisedSubstring?.summary_text??'').split(/s+/).slice(targetSummarisedSubstringTokenCount).join(' ')+contentAfter;
				if(this.DEBUG)console.log('Summarised candidate: '+summarisedString);
				return summarisedString;
			}))).filter(result=>result.status==='fulfilled'&&result.value).map(result=>(result as PromiseFulfilledResult<string>).value);
		}else{
			summaries=queryResult.map(result=>result.pageContent);
		}

		let rankingPromptTemplate=this.generatePromptTemplate(generateRankingInstruction({
			problemDescription,
			summaries,
			candidatesForFinalSelection,
			candidateIdentifierField:String(candidateIdentifierField)
		}));
		if(this.DEBUG)console.log('Formatted ranking prompt: '+rankingPromptTemplate);
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
		if(this.DEBUG)console.log('Generated rationale: '+rationale);
		let rationaleResponseIndex=rationale.indexOf('### Response:');
		if(rationaleResponseIndex>=0)rationaleResponseIndex+='### Response:'.length;
		else rationaleResponseIndex=0;
		rationale=rationale.substring(rationaleResponseIndex);
		//if(!rationale)throw new Error('No rationale generated');

		if(rationale){
			let identifiers=extractIdentifiersFromRationale(rationale);
			if(identifiers.length>candidatesForFinalSelection)identifiers=identifiers.slice(0,candidatesForFinalSelection);
			selectedCandidates=lodash.compact(identifiers.map(identifier=>{
				let selectedCandidate=candidates.find(candidate=>String(candidate[candidateIdentifierField]).toLowerCase()===identifier.toLowerCase());
				if(selectedCandidate)return selectedCandidate;
				selectedCandidate=candidates.find(candidate=>String(candidate[candidateIdentifierField]).toLowerCase().includes(identifier.toLowerCase()));
				if(selectedCandidate)return selectedCandidate;
				selectedCandidate=candidates.find(candidate=>identifier.toLowerCase().includes(String(candidate[candidateIdentifierField]).toLowerCase()));
				if(selectedCandidate)return selectedCandidate;
				return null;
			}));
		}

		if(!Array.isArray(selectedCandidates)||selectedCandidates.length<=0){
			selectedCandidates=lodash.uniq(lodash.compact(queryResult.map(result=>{
				let identifier=extractIdentifierFromCandidateDocument({
					candidateDocument:result.pageContent,
					candidateIdentifierField:String(candidateIdentifierField)
				})
				let selectedCandidate=candidates.find(candidate=>String(candidate[candidateIdentifierField]).toLowerCase()===identifier.toLowerCase());
				if(selectedCandidate)return selectedCandidate;
				selectedCandidate=candidates.find(candidate=>String(candidate[candidateIdentifierField]).toLowerCase().includes(identifier.toLowerCase()));
				if(selectedCandidate)return selectedCandidate;
				selectedCandidate=candidates.find(candidate=>identifier.toLowerCase().includes(String(candidate[candidateIdentifierField]).toLowerCase()));
				if(selectedCandidate)return selectedCandidate;
				return null;
			}))).slice(candidatesForFinalSelection);
		}
		if(this.DEBUG)console.log('Selected candidates',selectedCandidates);

		return{
			rationale,
			selectedCandidates
		};
	}
};

export namespace AICompareCandidates{
	export interface LoadArguments{
		progressCallback?:ProgressCallback;
		modelName:string;
	};

	export interface SummarisableSubstringIndices{
		start:number;
		end:number;
	};

	export interface CompareArguments<Candidate>{
		candidates:Candidate[];
		problemDescription:string;
		generateSearchAreasInstruction?:(problemDescription:string)=>string;
		convertCandidateToDocument?:(convertCandidateToDocumentArguments:ConvertCandidateToDocumentArguments<Candidate>)=>string;
		candidatesForInitialSelection?:number;
		candidatesForFinalSelection?:number;
		generateRankingInstruction?:(generateRankingInstructionArguments:GenerateRankingInstructionArguments)=>string;
		extractIdentifiersFromRationale?:(rationale:string)=>string[];
		extractIdentifierFromCandidateDocument?:(extractIdentifierFromCandidateDocumentArguments:ExtractIdentifierFromCandidateDocumentArguments)=>string;
		candidateIdentifierField?:keyof Candidate;
		getSummarisableSubstringIndices?:(candidateDocument:string)=>SummarisableSubstringIndices;
	};

	export interface ConvertCandidateToDocumentArguments<Candidate>{
		candidate:Candidate;
		index:number;
	};

	export interface ExtractIdentifierFromCandidateDocumentArguments{
		candidateDocument:string;
		candidateIdentifierField:string;
	};

	export interface GenerateRankingInstructionArguments{
		problemDescription:string;
		summaries:string[];
		candidatesForFinalSelection:number;
		candidateIdentifierField:string;
	};

	export interface CompareCandidatesReturn<Candidate>{
		selectedCandidates:Candidate[],
		rationale:string
	};
};

export default AICompareCandidates;