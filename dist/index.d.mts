import { ProgressCallback, ProgressInfo, TextGenerationPipeline } from "@huggingface/transformers";

//#region src/index.d.ts
declare class AICompareCandidates {
  generatorModelName: string;
  generatorPromise: Promise<TextGenerationPipeline> | null;
  generatorProgressInfo: ProgressInfo | null;
  generatorProgressCallback: ProgressCallback | null;
  loadGenerator(generatorProgressCallback?: ProgressCallback, generatorModelName?: string): Promise<void>;
}
//#endregion
export { AICompareCandidates, AICompareCandidates as default };