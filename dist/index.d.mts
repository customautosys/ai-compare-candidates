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
  }?: {
    progressCallback?: ProgressCallback;
    modelName: string;
  }): Promise<TextGenerationPipeline>;
  checkGeneratorLoaded(): Promise<void>;
  loadSummariser({
    progressCallback,
    modelName
  }?: {
    progressCallback?: ProgressCallback;
    modelName: string;
  }): Promise<SummarizationPipeline>;
  checkSummariserLoaded(): Promise<void>;
  loadEmbedder({
    progressCallback,
    modelName
  }?: {
    progressCallback?: ProgressCallback;
    modelName: string;
  }): Promise<FeatureExtractionPipeline>;
  checkEmbedderLoaded(): Promise<void>;
  loadTokeniser({
    progressCallback,
    modelName
  }?: {
    progressCallback?: ProgressCallback;
    modelName: string;
  }): Promise<PreTrainedTokenizer>;
  checkTokeniserLoaded(): Promise<void>;
  embedQuery(text: string): Promise<number[]>;
  embedDocuments(texts: string[]): Promise<number[][]>;
  generatePromptTemplate(prompt: string): string;
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
  }: {
    candidates: Candidate[];
    problemDescription: string;
    generateSearchAreasInstruction: (problemDescription: string) => string;
    convertCandidateToDocument: (candidate: Candidate) => string;
    candidatesForInitialSelection: number;
    candidatesForFinalSelection: number;
    generateRankingInstruction: (problemDescription: string, summaries: string[]) => string;
    extractIdentifiersFromRationale: (rationale: string) => string[];
    candidateIdentifierField: keyof Candidate;
    getSummarisableSubstringIndices?: (candidateDocument: string) => {
      start: number;
      end: number;
    };
  }): Promise<{
    selectedCandidates: Candidate[];
    rationale: string;
  }>;
}
//#endregion
export { AICompareCandidates, AICompareCandidates as default };