import { FeatureExtractionPipeline, PreTrainedTokenizer, ProgressCallback, ProgressInfo, SummarizationPipeline, TextGenerationPipeline } from "@huggingface/transformers";
import { Embeddings } from "@langchain/core/embeddings";
import { TransformersEnvironment } from "@huggingface/transformers/types/env";

//#region src/index.d.ts
declare class AICompareCandidates extends Embeddings {
  readonly env: TransformersEnvironment;
  DEBUG: boolean;
  generator: TextGenerationPipeline | null;
  generatorModelName: string;
  generatorPromise: Promise<TextGenerationPipeline> | null;
  generatorProgressInfo: ProgressInfo;
  generatorProgressCallback: ProgressCallback | null;
  summariser: SummarizationPipeline | null;
  summariserModelName: string;
  summariserPromise: Promise<SummarizationPipeline> | null;
  summariserProgressInfo: ProgressInfo;
  summariserProgressCallback: ProgressCallback | null;
  embedder: FeatureExtractionPipeline | null;
  embedderModelName: string;
  embedderPromise: Promise<FeatureExtractionPipeline> | null;
  embedderProgressInfo: ProgressInfo;
  embedderProgressCallback: ProgressCallback | null;
  tokeniser: PreTrainedTokenizer | null;
  tokeniserModelName: string;
  tokeniserPromise: Promise<PreTrainedTokenizer> | null;
  tokeniserProgressInfo: ProgressInfo;
  tokeniserProgressCallback: ProgressCallback | null;
  generateSearchAreasMaxNewTokens: number;
  generateSearchAreasTemperature: number;
  generateSearchAreasRepetitionPenalty: number;
  rankingMaxNewTokens: number;
  rankingTemperature: number;
  rankingRepetitionPenalty: number;
  targetSummarisedStringTokenCount: number;
  loadGenerator({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<TextGenerationPipeline>;
  checkGeneratorLoaded(): Promise<void>;
  loadSummariser({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<SummarizationPipeline>;
  checkSummariserLoaded(): Promise<void>;
  loadEmbedder({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<FeatureExtractionPipeline>;
  checkEmbedderLoaded(): Promise<void>;
  loadTokeniser({
    progressCallback,
    modelName
  }?: AICompareCandidates.LoadArguments): Promise<PreTrainedTokenizer>;
  checkTokeniserLoaded(): Promise<void>;
  embedQuery(text: string): Promise<number[]>;
  embedDocuments(texts: string[]): Promise<number[][]>;
  generatePromptTemplate(prompt: string): string;
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
    candidateIdentifierField,
    getSummarisableSubstringIndices
  }?: AICompareCandidates.CompareArguments<Candidate>): Promise<{
    selectedCandidates: Candidate[];
    rationale: string;
  }>;
}
declare namespace AICompareCandidates {
  interface LoadArguments {
    progressCallback?: ProgressCallback;
    modelName: string;
  }
  interface SummarisableSubstringIndices {
    start: number;
    end: number;
  }
  interface CompareArguments<Candidate> {
    candidates: Candidate[];
    problemDescription: string;
    generateSearchAreasInstruction: (problemDescription: string) => string;
    convertCandidateToDocument: (convertCandidateToDocumentArguments: ConvertCandidateToDocumentArguments<Candidate>) => string;
    candidatesForInitialSelection: number;
    candidatesForFinalSelection: number;
    generateRankingInstruction: (generateRankingInstructionArguments: GenerateRankingInstructionArguments) => string;
    extractIdentifiersFromRationale: (rationale: string) => string[];
    candidateIdentifierField: keyof Candidate;
    getSummarisableSubstringIndices?: (candidateDocument: string) => SummarisableSubstringIndices;
  }
  interface ConvertCandidateToDocumentArguments<Candidate> {
    candidate: Candidate;
    index: number;
  }
  interface GenerateRankingInstructionArguments {
    problemDescription: string;
    summaries: string[];
    candidatesForFinalSelection: number;
    candidateIdentifierField: string;
  }
}
//#endregion
export { AICompareCandidates, AICompareCandidates as default };