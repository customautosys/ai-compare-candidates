import * as _sroussey_transformers_types_env0 from "@sroussey/transformers/types/env";
import { FeatureExtractionPipeline, PreTrainedTokenizer, ProgressCallback, ProgressInfo, SummarizationPipeline, TextGenerationPipeline } from "@sroussey/transformers";
import { Embeddings } from "@langchain/core/embeddings";

//#region src/index.d.ts
declare class AICompareCandidates extends Embeddings {
  readonly env: _sroussey_transformers_types_env0.TransformersEnvironment;
  DEBUG: boolean;
  generator: TextGenerationPipeline | null;
  generatorModelName: string;
  generatorPromise: Promise<TextGenerationPipeline> | null;
  generatorProgressInfo: ProgressInfo;
  generatorProgressCallback: ProgressCallback | null;
  generatorAbortController: AbortController;
  summariser: SummarizationPipeline | null;
  summariserModelName: string;
  summariserPromise: Promise<SummarizationPipeline> | null;
  summariserProgressInfo: ProgressInfo;
  summariserProgressCallback: ProgressCallback | null;
  summariserAbortController: AbortController;
  embedder: FeatureExtractionPipeline | null;
  embedderModelName: string;
  embedderPromise: Promise<FeatureExtractionPipeline> | null;
  embedderProgressInfo: ProgressInfo;
  embedderProgressCallback: ProgressCallback | null;
  embedderAbortController: AbortController;
  tokeniser: PreTrainedTokenizer | null;
  tokeniserModelName: string;
  tokeniserPromise: Promise<PreTrainedTokenizer> | null;
  tokeniserProgressInfo: ProgressInfo;
  tokeniserProgressCallback: ProgressCallback | null;
  tokeniserAbortController: AbortController;
  generateSearchAreasMaxNewTokens: number;
  generateSearchAreasTemperature: number;
  generateSearchAreasRepetitionPenalty: number;
  rankingMaxNewTokens: number;
  rankingTemperature: number;
  rankingRepetitionPenalty: number;
  targetSummarisedStringTokenCount: number;
  constructor();
  loadGenerator({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<TextGenerationPipeline>;
  checkGeneratorLoaded({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<void>;
  abortLoadGenerator(reason?: any): Promise<void>;
  loadSummariser({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<SummarizationPipeline>;
  checkSummariserLoaded({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<void>;
  abortLoadSummariser(): Promise<void>;
  loadEmbedder({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<FeatureExtractionPipeline>;
  checkEmbedderLoaded({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<void>;
  abortLoadEmbedder(): Promise<void>;
  loadTokeniser({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<PreTrainedTokenizer>;
  checkTokeniserLoaded({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<void>;
  abortLoadTokeniser(): Promise<void>;
  embedQuery(text: string): Promise<number[]>;
  embedDocuments(texts: string[]): Promise<number[][]>;
  defaultGeneratePromptTemplate(prompt: string): string;
  defaultGenerateSearchAreasInstruction(problemDescription: string): string;
  defaultConvertCandidateToDocument<Candidate>({
    candidate,
    index
  }?: AICompareCandidates.ConvertCandidateToDocumentArguments<Candidate>): string;
  defaultGenerateRankingInstruction({
    problemDescription,
    summaries,
    candidatesForFinalSelection,
    candidateIdentifierField
  }?: AICompareCandidates.GenerateRankingInstructionArguments): string;
  regexIndexOf(text: string, regex: RegExp, startIndex: number): number;
  defaultExtractIdentifierFromCandidateDocument({
    candidateDocument,
    candidateIdentifierField
  }?: AICompareCandidates.ExtractIdentifierFromCandidateDocumentArguments): string;
  defaultExtractIdentifiersFromRationale(rationale: string): string[];
  compareCandidates<Candidate>({
    candidates,
    problemDescription,
    generateSearchAreasInstruction,
    convertCandidateToDocument,
    candidatesForInitialSelection,
    candidatesForFinalSelection,
    generateRankingInstruction,
    extractIdentifiersFromRationale,
    extractIdentifierFromCandidateDocument,
    candidateIdentifierField,
    getSummarisableSubstringIndices,
    generatePromptTemplate
  }?: AICompareCandidates.CompareArguments<Candidate>): Promise<AICompareCandidates.CompareCandidatesReturn<Candidate> | void>;
}
declare namespace AICompareCandidates {
  interface LoadArguments {
    progressCallback?: ProgressCallback;
    modelName?: string;
  }
  interface SummarisableSubstringIndices {
    start: number;
    end: number;
  }
  interface CompareArguments<Candidate> {
    candidates: Candidate[];
    problemDescription: string;
    generateSearchAreasInstruction?: (problemDescription: string) => string;
    convertCandidateToDocument?: (convertCandidateToDocumentArguments: ConvertCandidateToDocumentArguments<Candidate>) => string;
    candidatesForInitialSelection?: number;
    candidatesForFinalSelection?: number;
    generateRankingInstruction?: (generateRankingInstructionArguments: GenerateRankingInstructionArguments) => string;
    extractIdentifiersFromRationale?: (rationale: string) => string[];
    extractIdentifierFromCandidateDocument?: (extractIdentifierFromCandidateDocumentArguments: ExtractIdentifierFromCandidateDocumentArguments) => string;
    candidateIdentifierField?: keyof Candidate;
    getSummarisableSubstringIndices?: (candidateDocument: string) => SummarisableSubstringIndices;
    generatePromptTemplate?: (prompt: string) => string;
  }
  interface ConvertCandidateToDocumentArguments<Candidate> {
    candidate: Candidate;
    index: number;
  }
  interface ExtractIdentifierFromCandidateDocumentArguments {
    candidateDocument: string;
    candidateIdentifierField: string;
  }
  interface GenerateRankingInstructionArguments {
    problemDescription: string;
    summaries: string[];
    candidatesForFinalSelection: number;
    candidateIdentifierField: string;
  }
  interface CompareCandidatesReturn<Candidate> {
    selectedCandidates: Candidate[];
    rationale: string;
  }
}
//#endregion
export { AICompareCandidates, AICompareCandidates as default };
//# sourceMappingURL=index.d.cts.map