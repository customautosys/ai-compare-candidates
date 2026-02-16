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
	TextGenerationConfig,
	TextGenerationSingle,
	TextStreamer
}from '@sroussey/transformers';
import {MemoryVectorStore} from '@langchain/classic/vectorstores/memory';
import {Embeddings} from '@langchain/core/embeddings';
import lodash from 'lodash';
import jsan from 'jsan';

export class AICompareCandidates extends Embeddings{
	readonly env=env;
	DEBUG=true;

	generator:TextGenerationPipeline|null=null;
	generatorModelName='Xenova/Phi-3-mini-4k-instruct';
	generatorPromise:Promise<TextGenerationPipeline>|null=null;
	generatorProgressInfo:ProgressInfo=<ProgressInfo>{};
	generatorProgressCallback:ProgressCallback|null=null;
	generatorAbortController=new AbortController();

	summariser:SummarizationPipeline|null=null;
	summariserModelName='Xenova/distilbart-cnn-12-6';
	summariserPromise:Promise<SummarizationPipeline>|null=null;
	summariserProgressInfo:ProgressInfo=<ProgressInfo>{};
	summariserProgressCallback:ProgressCallback|null=null;
	summariserAbortController=new AbortController();

	embedder:FeatureExtractionPipeline|null=null;
	embedderModelName='Xenova/all-MiniLM-L12-v2';
	embedderPromise:Promise<FeatureExtractionPipeline>|null=null;
	embedderProgressInfo:ProgressInfo=<ProgressInfo>{};
	embedderProgressCallback:ProgressCallback|null=null;
	embedderAbortController=new AbortController();

	tokeniser:PreTrainedTokenizer|null=null;
	tokeniserModelName=this.generatorModelName;
	tokeniserPromise:Promise<PreTrainedTokenizer>|null=null;
	tokeniserProgressInfo:ProgressInfo=<ProgressInfo>{};
	tokeniserProgressCallback:ProgressCallback|null=null;
	tokeniserAbortController=new AbortController();

	generateSearchAreasMaxNewTokens=64;
	generateSearchAreasTemperature=0.35;
	generateSearchAreasRepetitionPenalty=1.5;

	rankingMaxNewTokens=64;
	rankingTemperature=0.35;
	rankingRepetitionPenalty=1.5;

	targetSummarisedStringTokenCount=300;

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
			},
			abort_signal:this.generatorAbortController.signal
		});
		this.generator=await this.generatorPromise;
		return this.generator;
	}

	async checkGeneratorLoaded({
		progressCallback,
		modelName=''
	}:AICompareCandidates.LoadArguments=<AICompareCandidates.LoadArguments>{}){
		if(!this.generatorPromise)this.loadGenerator({
			progressCallback,
			modelName
		});
		if(!this.generator){
			try{
				await this.generatorPromise;
			}catch(error){
				this.generatorPromise=null;
				throw error;
			}
		}
		if(!this.generator){
			this.generatorPromise=null;
			throw new Error('Unable to load generator');
		}
	}

	async abortLoadGenerator(reason?:any){
		this.generatorAbortController.abort(reason);
		this.generatorAbortController=new AbortController();
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
			},
			abort_signal:this.summariserAbortController.signal
		});
		this.summariser=await this.summariserPromise;
		return this.summariser;
	}

	async checkSummariserLoaded({
		progressCallback,
		modelName=''
	}:AICompareCandidates.LoadArguments=<AICompareCandidates.LoadArguments>{}){
		if(!this.summariserPromise)this.loadSummariser({
			progressCallback,
			modelName
		});
		if(!this.summariser){
			try{
				await this.summariserPromise;
			}catch(error){
				this.summariserPromise=null;
				throw error;
			}
		}
		if(!this.summariser){
			this.summariserPromise=null;
			throw new Error('Unable to load summariser');
		}
	}

	async abortLoadSummariser(){
		this.summariserAbortController.abort();
		this.summariserAbortController=new AbortController();
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
			},
			abort_signal:this.embedderAbortController.signal
		});
		this.embedder=await this.embedderPromise;
		return this.embedder;
	}

	async checkEmbedderLoaded({
		progressCallback,
		modelName=''
	}:AICompareCandidates.LoadArguments=<AICompareCandidates.LoadArguments>{}){
		if(!this.embedderPromise)this.loadEmbedder({
			progressCallback,
			modelName
		});
		if(!this.embedder){
			try{
				await this.embedderPromise;
			}catch(error){
				this.embedderPromise=null;
				throw error;
			}
		}
		if(!this.embedder){
			this.embedderPromise=null;
			throw new Error('Unable to load embedder');
		}
	}

	async abortLoadEmbedder(){
		this.embedderAbortController.abort();
		this.embedderAbortController=new AbortController();
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
			},
			abort_signal:this.tokeniserAbortController.signal
		});
		this.tokeniser=await this.tokeniserPromise;
		return this.tokeniser;
	}

	async checkTokeniserLoaded({
		progressCallback,
		modelName=''
	}:AICompareCandidates.LoadArguments=<AICompareCandidates.LoadArguments>{}){
		if(!this.tokeniserPromise)this.loadTokeniser({
			progressCallback,
			modelName
		});
		if(!this.tokeniser){
			try{
				await this.tokeniserPromise;
			}catch(error){
				this.tokeniserPromise=null;
				throw error;
			}
		}
		if(!this.tokeniser){
			this.tokeniserPromise=null;
			throw new Error('Unable to load tokeniser');
		}
	}

	async abortLoadTokeniser(){
		this.tokeniserAbortController.abort();
		this.tokeniserAbortController=new AbortController();
	}

	async embedQuery(text:string):Promise<number[]>{
		await this.checkEmbedderLoaded();
		if(typeof this.embedder==='function')return Array.from((await this.embedder(text,{
			pooling:'mean',
			normalize:true
		}))?.data);
		return [];
	}

	async embedDocuments(texts:string[]):Promise<number[][]>{
		return Promise.all(texts.map(text=>this.embedQuery(text)));
	}

	defaultGeneratePromptTemplate(prompt:string){
		return prompt;
	}

	async defaultPerformGeneration({
		generator,
		tokeniser,
		prompt='',
		textGenerationConfig={}
	}:AICompareCandidates.PerformGenerationArguments=<AICompareCandidates.PerformGenerationArguments>{}){
		let errors='';
		if(!generator)errors+='No generator provided.';
		if(!tokeniser)errors+=(errors?'\n':'')+'No tokeniser provided.';
		if(typeof prompt!=='string'||!prompt)errors+=(errors?'\n':'')+'No valid prompt provided.';
		if(errors)throw new Error(errors);
		let messages=[{
			role:'user',
			content:prompt
		}];
		let streamer=new TextStreamer(tokeniser,{
			skip_prompt:true
		});
		if(!textGenerationConfig.pad_token_id)textGenerationConfig.pad_token_id=tokeniser.pad_token_id??tokeniser.sep_token_id??0;
		if(!textGenerationConfig.eos_token_id)textGenerationConfig.eos_token_id=tokeniser.sep_token_id??2;
		let output=await generator(messages,textGenerationConfig);
		if(this.DEBUG)console.log(output);
		let outputObject=Array.isArray(output?.[0])?output?.[0]?.[0]:output?.[0];
		if(!outputObject.generated_text)throw new Error('No generated text');
		if(typeof outputObject.generated_text==='string')return outputObject.generated_text;
		let content=outputObject.generated_text?.at(-1)?.content;
		if(typeof content!=='string'||!content)throw new Error('No content in generated text');
		return content;
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
		return 'Rank only the top '+candidatesForFinalSelection+' candidates by connecting them to "'+problemDescription+'" stating # followed by the candidate number and the full '+lodash.startCase(candidateIdentifierField)+' followed by a 15-word explanation for each ranking. Stop generating text immediately thereafter.\n\nCandidates:\n\n'+summaries.join('\n\n');
	}

	regexIndexOf(text:string,regex:RegExp,startIndex:number){
    	let indexInSuffix=text.slice(startIndex).search(regex);
    	return indexInSuffix<0?indexInSuffix:indexInSuffix+startIndex;
	}

	defaultExtractIdentifierFromCandidateDocument({
		candidateDocument,
		candidateIdentifierField
	}:AICompareCandidates.ExtractIdentifierFromCandidateDocumentArguments=<AICompareCandidates.ExtractIdentifierFromCandidateDocumentArguments>{}){
		if(this.DEBUG)console.log(candidateDocument,candidateIdentifierField);
		let startCase=lodash.startCase(candidateIdentifierField);
		let startIndex=candidateDocument.indexOf(startCase);
		if(startIndex<0)startIndex=candidateDocument.toLowerCase().indexOf(startCase.toLowerCase());
		if(this.DEBUG)console.log(startIndex);
		if(startIndex>=0)startIndex+=startCase.length;
		if(startIndex<0){
			startIndex=candidateDocument.toLowerCase().indexOf(candidateIdentifierField.toLowerCase());
			if(startIndex>=0)startIndex+=candidateIdentifierField.length;
		}
		if(this.DEBUG)console.log(startIndex);
		else return '';
		startIndex=candidateDocument.indexOf(':',startIndex);
		if(this.DEBUG)console.log(startIndex);
		if(startIndex<0)startIndex=this.regexIndexOf(candidateDocument,/\s+/,startIndex);
		if(this.DEBUG)console.log(startIndex);
		if(startIndex<0)return '';
		let endIndex=candidateDocument.indexOf('\n',startIndex);
		if(endIndex<0)endIndex=candidateDocument.length;
		if(this.DEBUG)console.log(endIndex);
		return candidateDocument.substring(startIndex,endIndex).trim();
	}

	defaultExtractIdentifiersFromRationale(rationale:string){
		let regex=/^\d+?\s*?\p{P}\s*?(.+?)\s*?(-|\.|;|\n)/gmu;
		let matches:string[]=[];
		for(let match:RegExpExecArray|null;Array.isArray(match=regex.exec(rationale));)if(match[1])matches.push(match[1]);
		return matches;
	}

	defaultFindCandidateFromIdentifier<Candidate>({
		identifier,
		candidateIdentifierField,
		candidates
	}:AICompareCandidates.FindCandidateFromIdentifierArguments<Candidate>=<AICompareCandidates.FindCandidateFromIdentifierArguments<Candidate>>{}){
		if(!identifier||typeof identifier!=='string')identifier=String(identifier);
		console.log('identifier',identifier);
		let selectedCandidate=candidates.find(candidate=>String(candidate[candidateIdentifierField])===identifier);
		if(selectedCandidate){
			if(this.DEBUG){
				console.log('Candidate found based on case-sensitive match',selectedCandidate[candidateIdentifierField]);
			}
			return selectedCandidate;
		}
		selectedCandidate=candidates.find(candidate=>String(candidate[candidateIdentifierField]).toLowerCase()===identifier.toLowerCase());
		if(selectedCandidate){
			if(this.DEBUG){
				console.log('Candidate found based on case-sensitive match',selectedCandidate[candidateIdentifierField]);
			}
			return selectedCandidate;
		}
		selectedCandidate=candidates.find(candidate=>String(candidate[candidateIdentifierField]).toLowerCase().includes(identifier.toLowerCase()));
		if(selectedCandidate){
			if(this.DEBUG){
				console.log('Candidate found based on case-sensitive match',selectedCandidate[candidateIdentifierField]);
			}
			return selectedCandidate;
		}
		selectedCandidate=candidates.find(candidate=>identifier.toLowerCase().includes(String(candidate[candidateIdentifierField]).toLowerCase()));
		if(selectedCandidate){
			if(this.DEBUG){
				console.log('Candidate found based on case-sensitive match',selectedCandidate[candidateIdentifierField]);
			}
			return selectedCandidate;
		}
		//split by space and find highest number of matches (tie break if it is in same order)
		let identifierWords=identifier.split(/\s+/g).filter(identifierWord=>/\w/.test(identifierWord));
		if(this.DEBUG)console.log('Identifier words',identifierWords);
		let selectedCandidates=candidates.map(candidate=>({
			identifierWordIndices:identifierWords.map(identifierWord=>String(candidate[candidateIdentifierField]).indexOf(identifierWord)),
			candidate
		})).filter(candidate=>candidate.identifierWordIndices.some(index=>index>=0));
		if(selectedCandidates.length>1){
			selectedCandidates.sort((a,b)=>{
				let aCount=lodash.sumBy(a.identifierWordIndices,aElement=>aElement<0?0:1);
				let bCount=lodash.sumBy(b.identifierWordIndices,bElement=>bElement<0?0:1);
				if(this.DEBUG){
					console.log('a',a);
					console.log('b',b);
					console.log('a.identifierWordIndices',a.identifierWordIndices);
					console.log('b.identifierWordIndices',b.identifierWordIndices);
					console.log('aCount',aCount);
					console.log('bCount',bCount);
				}
				if(aCount!==bCount)return bCount-aCount;
				if(aCount===0&&bCount===0)return 0;
				aCount=0;
				bCount=0;
				for(let i=0;i<a.identifierWordIndices.length-1;++i){
					for(let j=i+1;j<a.identifierWordIndices.length;++j){
						if(a.identifierWordIndices[i]<0||a.identifierWordIndices[j]<0)continue;
						if(a.identifierWordIndices[i]<a.identifierWordIndices[j]){
							if(this.DEBUG){
								console.log('i',i);
								console.log('j',j);
								console.log('a.identifierWordIndices[i]',a.identifierWordIndices[i]);
								console.log('a.identifierWordIndices[j]',a.identifierWordIndices[j]);
							}
							++aCount;
						}
					}
				}
				for(let i=0;i<b.identifierWordIndices.length-1;++i){
					for(let j=i+1;j<b.identifierWordIndices.length;++j){
						if(b.identifierWordIndices[i]<0||b.identifierWordIndices[j]<0)continue;
						if(b.identifierWordIndices[i]<b.identifierWordIndices[j]){
							if(this.DEBUG){
								console.log('i',i);
								console.log('j',j);
								console.log('b.identifierWordIndices[i]',b.identifierWordIndices[i]);
								console.log('b.identifierWordIndices[j]',b.identifierWordIndices[j]);
							}
							++bCount;
						}
					}
				}
				if(this.DEBUG){
					console.log('aCount',aCount);
					console.log('bCount',bCount);
				}
				return bCount-aCount;
			});
		}
		if(selectedCandidates.length>=1&&selectedCandidates[0].candidate)return selectedCandidates[0].candidate;
		return null;
	}

	defaultParseSearchAreasResponse(searchAreasResponse:string){
		let searchAreasResponseIndex=String(searchAreasResponse).indexOf('### Response:');
		if(searchAreasResponseIndex>=0)searchAreasResponseIndex+='### Response:'.length;
		else searchAreasResponseIndex=0;
		return String(searchAreasResponse).substring(searchAreasResponseIndex).trim();
	}

	errorMessage(error:any){
		return typeof error?.response?.data==='string'?error.response.data:error?.response?.data?jsan.stringify(error.response.data):typeof error?.message==='string'?error.message:error?.message?jsan.stringify(error.message):typeof error==='string'?error:jsan.stringify(error);
	}

	async compareCandidates<Candidate>({
		candidates,
		problemDescription='',
		performGeneration=this.defaultPerformGeneration.bind(this),
		generateSearchAreasInstruction=this.defaultGenerateSearchAreasInstruction.bind(this),
		parseSearchAreasResponse=this.defaultParseSearchAreasResponse.bind(this),
		convertCandidateToDocument=this.defaultConvertCandidateToDocument.bind(this),
		candidatesForInitialSelection=2,
		candidatesForFinalSelection=1,
		generateRankingInstruction=this.defaultGenerateRankingInstruction.bind(this),
		extractIdentifiersFromRationale=this.defaultExtractIdentifiersFromRationale.bind(this),
		extractIdentifierFromCandidateDocument=this.defaultExtractIdentifierFromCandidateDocument.bind(this),
		candidateIdentifierField=undefined,
		findCandidateFromIdentifier=this.defaultFindCandidateFromIdentifier.bind(this),
		getSummarisableSubstringIndices,
		generatePromptTemplate=this.defaultGeneratePromptTemplate.bind(this),
		skipRationale=false,
		parallelSummarisation=true
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

		let searchAreasPromptTemplate=generatePromptTemplate(generateSearchAreasInstruction(problemDescription));
		if(this.DEBUG)console.log('Formatted search areas prompt: '+searchAreasPromptTemplate);
		await this.checkTokeniserLoaded();
		if(!this.tokeniser)return;
		let searchAreasPromptTokens=this.tokeniser.encode(searchAreasPromptTemplate);
		if(searchAreasPromptTokens.length>this.tokeniser.model_max_length)throw new Error('Search areas instruction prompt is too long for the tokeniser model');

		await this.checkGeneratorLoaded();
		if(!this.generator)return;
		let generatedText=await performGeneration({
			generator:this.generator,
			tokeniser:this.tokeniser,
			prompt:searchAreasPromptTemplate,
			textGenerationConfig:{
				max_new_tokens:this.generateSearchAreasMaxNewTokens,
				temperature:this.generateSearchAreasTemperature,
				repetition_penalty:this.generateSearchAreasRepetitionPenalty
			}
		});
		if(this.DEBUG)console.log('Generated search areas response: '+generatedText);
		let vectorSearchQuery=parseSearchAreasResponse(generatedText);
		//generally the first sentence has the greatest relevance to the actual prompt
		//if(vectorSearchQuery.includes('.'))vectorSearchQuery=vectorSearchQuery.split('.')[0].trim();
		if(this.DEBUG)console.log('Vector search query: '+vectorSearchQuery);
		let queryResult=await vectorStore.similaritySearch(vectorSearchQuery,candidatesForInitialSelection);
		if(this.DEBUG)console.log('Vector search results: ',queryResult);

		let summaries:string[]=[];
		//only bother doing summarisation if there are candidates which exceed the token count
		if(queryResult.some(result=>result.pageContent.trim().split(/\s+/).length>this.targetSummarisedStringTokenCount)){
			await this.checkSummariserLoaded();
			if(!this.summariser)return;
			if(parallelSummarisation){
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
					if(this.DEBUG)console.log(summarisableSubstringIndices);
					let summarisableSubstring=result.pageContent.substring(summarisableSubstringIndices.start,summarisableSubstringIndices.end);
					if(this.DEBUG)console.log(summarisableSubstring);
					let contentBefore=result.pageContent.substring(0,summarisableSubstringIndices.start);
					let contentAfter=result.pageContent.substring(summarisableSubstringIndices.end);
					let wordsWithoutSummarisable=contentBefore.split(/\s+/).length+contentAfter.split(/\s+/).length;
					let targetSummarisedSubstringTokenCount=Math.max(1,this.targetSummarisedStringTokenCount-wordsWithoutSummarisable);
					if(this.DEBUG)console.log(wordsWithoutSummarisable,targetSummarisedSubstringTokenCount);
					if(!this.summariser)return contentBefore+contentAfter;
					let summarisedSubstringArray=await this.summariser(summarisableSubstring,<TextGenerationConfig>{
						max_length:targetSummarisedSubstringTokenCount
					});
					let summarisedSubstring=Array.isArray(summarisedSubstringArray?.[0])?summarisedSubstringArray?.[0]?.[0]:summarisedSubstringArray?.[0];
					if(this.DEBUG)console.log(summarisedSubstringArray,summarisedSubstring,summarisedSubstring?.summary_text??'',(summarisedSubstring?.summary_text??'').split(/\s+/).slice(0,targetSummarisedSubstringTokenCount).join(' '));
					let summarisedString=contentBefore+(summarisedSubstring?.summary_text??'').split(/\s+/).slice(0,targetSummarisedSubstringTokenCount).join(' ').trim()+contentAfter;
					if(this.DEBUG)console.log('Summarised candidate: '+summarisedString);
					return summarisedString;
				}))).filter(result=>result.status==='fulfilled'&&result.value).map(result=>(result as PromiseFulfilledResult<string>).value);
			}else{
				for(let result of queryResult){
					try{
						if(!result.pageContent||typeof result.pageContent!=='string')continue;
						if(result.pageContent.trim().split(/\s+/).length<=this.targetSummarisedStringTokenCount){
							summaries.push(result.pageContent);
							continue;
						}
						let summarisableSubstringIndices:AICompareCandidates.SummarisableSubstringIndices={
							start:0,
							end:result.pageContent.length
						};
						if(getSummarisableSubstringIndices)Object.assign(summarisableSubstringIndices,getSummarisableSubstringIndices(result.pageContent));
						summarisableSubstringIndices.start=lodash.clamp(lodash.toSafeInteger(summarisableSubstringIndices.start),0,result.pageContent.length);
						summarisableSubstringIndices.end=lodash.clamp(lodash.toSafeInteger(summarisableSubstringIndices.end),0,result.pageContent.length);
						if(this.DEBUG)console.log(summarisableSubstringIndices);
						let summarisableSubstring=result.pageContent.substring(summarisableSubstringIndices.start,summarisableSubstringIndices.end);
						if(this.DEBUG)console.log(summarisableSubstring);
						let contentBefore=result.pageContent.substring(0,summarisableSubstringIndices.start);
						let contentAfter=result.pageContent.substring(summarisableSubstringIndices.end);
						let wordsWithoutSummarisable=contentBefore.split(/\s+/).length+contentAfter.split(/\s+/).length;
						let targetSummarisedSubstringTokenCount=Math.max(1,this.targetSummarisedStringTokenCount-wordsWithoutSummarisable);
						if(this.DEBUG)console.log(wordsWithoutSummarisable,targetSummarisedSubstringTokenCount);
						if(this.DEBUG)console.log(this.summariser,typeof this.summariser,this.summariser.call);
						let summarisedSubstringArray=await this.summariser(summarisableSubstring,<TextGenerationConfig>{
							max_length:targetSummarisedSubstringTokenCount
						});
						let summarisedSubstring=Array.isArray(summarisedSubstringArray?.[0])?summarisedSubstringArray?.[0]?.[0]:summarisedSubstringArray?.[0];
						if(this.DEBUG)console.log(summarisedSubstringArray,summarisedSubstring,summarisedSubstring?.summary_text??'',(summarisedSubstring?.summary_text??'').split(/\s+/).slice(0,targetSummarisedSubstringTokenCount).join(' '));
						let summarisedString=contentBefore+(summarisedSubstring?.summary_text??'').split(/\s+/).slice(0,targetSummarisedSubstringTokenCount).join(' ').trim()+contentAfter;
						if(this.DEBUG)console.log('Summarised candidate: '+summarisedString);
						summaries.push(summarisedString);
					}catch(error){
						console.log(error);
					}
				}
			}
		}else{
			summaries=queryResult.map(result=>result.pageContent);
		}
		
		let rationaleError='';

		if(!skipRationale){
			let rankingPromptTemplate=generatePromptTemplate(generateRankingInstruction({
				problemDescription,
				summaries,
				candidatesForFinalSelection,
				candidateIdentifierField:String(candidateIdentifierField)
			}));
			if(this.DEBUG)console.log('Formatted ranking prompt: '+rankingPromptTemplate);
			let rankingPromptTokens=this.tokeniser.encode(rankingPromptTemplate);
			if(this.DEBUG)console.log(rankingPromptTokens.length,this.tokeniser.model_max_length);
			if(rankingPromptTokens.length>this.tokeniser.model_max_length)throw new Error('Ranking instruction prompt is too long for the tokeniser model');
			try{
				let ranking=await performGeneration({
					generator:this.generator,
					tokeniser:this.tokeniser,
					prompt:rankingPromptTemplate,
					textGenerationConfig:{
						max_new_tokens:this.rankingMaxNewTokens,
						temperature:this.rankingTemperature,
						repetition_penalty:this.rankingRepetitionPenalty
					}
				});
				rationale=ranking.trim().replace(/(\*\*)|(<\/?s>)|(\[.*?\])\s*/g, '');
				if(this.DEBUG)console.log('Generated rationale: '+rationale);
				let rationaleResponseIndex=rationale.indexOf('### Response:');
				if(rationaleResponseIndex>=0)rationaleResponseIndex+='### Response:'.length;
				else rationaleResponseIndex=0;
				rationale=rationale.substring(rationaleResponseIndex);
				//if(!rationale)throw new Error('No rationale generated');
			}catch(error){
				console.log(error);
				rationaleError=this.errorMessage(error);
				rationale='';
			}
		}

		if(rationale){
			let identifiers=extractIdentifiersFromRationale(rationale);
			if(this.DEBUG)console.log('Extracted identifiers from rationale: '+identifiers);
			if(identifiers.length>candidatesForFinalSelection)identifiers=identifiers.slice(0,candidatesForFinalSelection);
			selectedCandidates=lodash.compact(identifiers.map(identifier=>findCandidateFromIdentifier({
				identifier,
				candidateIdentifierField,
				candidates
			})));
		}

		if(!Array.isArray(selectedCandidates)||selectedCandidates.length<candidatesForFinalSelection){
			if(!Array.isArray(selectedCandidates))selectedCandidates=[];
			let additionalSelectedCandidates=lodash.compact(queryResult.map(result=>{
				let identifier=extractIdentifierFromCandidateDocument({
					candidateDocument:result.pageContent,
					candidateIdentifierField:String(candidateIdentifierField)
				});
				if(this.DEBUG)console.log('Extracted identifier from candidate document: '+identifier);
				return findCandidateFromIdentifier({
					identifier,
					candidateIdentifierField,
					candidates
				});
			}));
			selectedCandidates.splice(selectedCandidates.length,0,...additionalSelectedCandidates);
			selectedCandidates=lodash.uniq(selectedCandidates).slice(0,candidatesForFinalSelection);
		}
		if(this.DEBUG)console.log('Selected candidates',selectedCandidates);

		return{
			rationale,
			summaries,
			...(rationaleError?{rationaleError}:{}),
			selectedCandidates
		};
	}
};

export namespace AICompareCandidates{
	export interface LoadArguments{
		progressCallback?:ProgressCallback;
		modelName?:string;
	};

	export interface SummarisableSubstringIndices{
		start:number;
		end:number;
	};

	export interface CompareArguments<Candidate>{
		candidates:Candidate[];
		problemDescription:string;
		performGeneration?:(performGenerationArguments:PerformGenerationArguments)=>Promise<string>;
		generateSearchAreasInstruction?:(problemDescription:string)=>string;
		parseSearchAreasResponse?:(searchAreasResponse:string)=>string;
		convertCandidateToDocument?:(convertCandidateToDocumentArguments:ConvertCandidateToDocumentArguments<Candidate>)=>string;
		candidatesForInitialSelection?:number;
		candidatesForFinalSelection?:number;
		generateRankingInstruction?:(generateRankingInstructionArguments:GenerateRankingInstructionArguments)=>string;
		extractIdentifiersFromRationale?:(rationale:string)=>string[];
		extractIdentifierFromCandidateDocument?:(extractIdentifierFromCandidateDocumentArguments:ExtractIdentifierFromCandidateDocumentArguments)=>string;
		candidateIdentifierField?:keyof Candidate;
		findCandidateFromIdentifier?:(findCandidateFromIdentifierArguments:FindCandidateFromIdentifierArguments<Candidate>)=>Candidate|null;
		getSummarisableSubstringIndices?:(candidateDocument:string)=>SummarisableSubstringIndices;
		generatePromptTemplate?:(prompt:string)=>string;
		skipRationale?:boolean;
		parallelSummarisation?:boolean;
	};

	export interface PerformGenerationArguments{
		generator:TextGenerationPipeline;
		tokeniser:PreTrainedTokenizer;
		prompt:string;
		textGenerationConfig?:Partial<TextGenerationConfig>;
	};

	export interface ConvertCandidateToDocumentArguments<Candidate>{
		candidate:Candidate;
		index:number;
	};

	export interface ExtractIdentifierFromCandidateDocumentArguments{
		candidateDocument:string;
		candidateIdentifierField:string;
	};

	export interface FindCandidateFromIdentifierArguments<Candidate>{
		identifier:string;
		candidateIdentifierField:keyof Candidate;
		candidates:Candidate[];
	};

	export interface GenerateRankingInstructionArguments{
		problemDescription:string;
		summaries:string[];
		candidatesForFinalSelection:number;
		candidateIdentifierField:string;
	};

	export interface CompareCandidatesReturn<Candidate>{
		selectedCandidates:Candidate[],
		rationale:string,
		rationaleError?:string,
		summaries:string[]
	};
};

export default AICompareCandidates;