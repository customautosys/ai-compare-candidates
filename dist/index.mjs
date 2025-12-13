import { AutoTokenizer, env, pipeline } from "@huggingface/transformers";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Embeddings } from "@langchain/core/embeddings";
import lodash from "lodash";

//#region src/index.ts
var AICompareCandidates = class extends Embeddings {
	constructor(..._args) {
		super(..._args);
		this.env = env;
		this.DEBUG = true;
		this.generator = null;
		this.generatorModelName = "Xenova/LaMini-GPT-774m";
		this.generatorPromise = null;
		this.generatorProgressInfo = {};
		this.generatorProgressCallback = null;
		this.summariser = null;
		this.summariserModelName = "Xenova/distilbart-cnn-12-6";
		this.summariserPromise = null;
		this.summariserProgressInfo = {};
		this.summariserProgressCallback = null;
		this.embedder = null;
		this.embedderModelName = "Xenova/all-MiniLM-L6-v2";
		this.embedderPromise = null;
		this.embedderProgressInfo = {};
		this.embedderProgressCallback = null;
		this.tokeniserModelName = "Xenova/LaMini-GPT-774m";
		this.tokeniserPromise = null;
		this.tokeniserProgressInfo = {};
		this.tokeniserProgressCallback = null;
		this.generateSearchAreasMaxNewTokens = 64;
		this.generateSearchAreasTemperature = .35;
		this.generateSearchAreasRepetitionPenalty = 1.5;
		this.rankingMaxNewTokens = 64;
		this.rankingTemperature = .35;
		this.rankingRepetitionPenalty = 1.5;
		this.targetSummarisedStringTokenCount = 420;
	}
	static {
		env.allowRemoteModels = true;
		env.allowLocalModels = true;
	}
	async loadGenerator({ progressCallback, modelName } = { modelName: "" }) {
		if (typeof modelName === "string" && modelName) this.generatorModelName = modelName;
		if (!this.generatorModelName) throw new Error("Invalid generator model name");
		if (progressCallback) this.generatorProgressCallback = progressCallback;
		this.generatorPromise = pipeline("text-generation", this.generatorModelName, {
			device: "webgpu",
			progress_callback: (progressInfo) => {
				Object.assign(this.generatorProgressInfo, progressInfo);
				return this.generatorProgressCallback?.(progressInfo);
			}
		});
		this.generator = await this.generatorPromise;
		return this.generator;
	}
	async checkGeneratorLoaded() {
		if (!this.generatorPromise) this.loadGenerator();
		if (!this.generator) await this.generatorPromise;
		if (!this.generator) throw new Error("Unable to load generator");
	}
	async loadSummariser({ progressCallback, modelName } = { modelName: "" }) {
		if (typeof modelName === "string" && modelName) this.summariserModelName = modelName;
		if (!this.summariserModelName) throw new Error("Invalid summariser model name");
		if (progressCallback) this.summariserProgressCallback = progressCallback;
		this.summariserPromise = pipeline("summarization", this.summariserModelName, {
			device: "webgpu",
			progress_callback: (progressInfo) => {
				Object.assign(this.summariserProgressInfo, progressInfo);
				return this.summariserProgressCallback?.(progressInfo);
			}
		});
		this.summariser = await this.summariserPromise;
		return this.summariser;
	}
	async checkSummariserLoaded() {
		if (!this.summariserPromise) this.loadEmbedder();
		if (!this.summariser) await this.summariserPromise;
		if (!this.summariser) throw new Error("Unable to load summariser");
	}
	async loadEmbedder({ progressCallback, modelName } = { modelName: "" }) {
		if (typeof modelName === "string" && modelName) this.embedderModelName = modelName;
		if (!this.embedderModelName) throw new Error("Invalid embedder model name");
		if (progressCallback) this.embedderProgressCallback = progressCallback;
		this.embedderPromise = pipeline("feature-extraction", this.embedderModelName, {
			device: "webgpu",
			progress_callback: (progressInfo) => {
				Object.assign(this.embedderProgressInfo, progressInfo);
				return this.embedderProgressCallback?.(progressInfo);
			}
		});
		this.embedder = await this.embedderPromise;
		return this.embedder;
	}
	async checkEmbedderLoaded() {
		if (!this.embedderPromise) this.loadEmbedder();
		if (!this.embedder) await this.embedderPromise;
		if (!this.embedder) throw new Error("Unable to load embedder");
	}
	async loadTokeniser({ progressCallback, modelName } = { modelName: "" }) {
		if (typeof modelName === "string" && modelName) this.tokeniserModelName = modelName;
		if (!this.tokeniserModelName) throw new Error("Invalid tokeniser model name");
		if (progressCallback) this.tokeniserProgressCallback = progressCallback;
		this.tokeniserPromise = AutoTokenizer.from_pretrained(this.tokeniserModelName, { progress_callback: (progressInfo) => {
			Object.assign(this.tokeniserProgressInfo, progressInfo);
			return this.tokeniserProgressCallback?.(progressInfo);
		} });
		this.tokeniser = await this.tokeniserPromise;
		return this.tokeniser;
	}
	async checkTokeniserLoaded() {
		if (!this.tokeniserPromise) this.loadTokeniser();
		if (!this.tokeniser) await this.tokeniserPromise;
		if (!this.tokeniser) throw new Error("Unable to load tokeniser");
	}
	async embedQuery(text) {
		await this.checkEmbedderLoaded();
		return Array.from((await this.embedder(text, {
			pooling: "mean",
			normalize: true
		})).data);
	}
	async embedDocuments(texts) {
		return Promise.all(texts.map((text) => this.embedQuery(text)));
	}
	generatePromptTemplate(prompt) {
		return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" + prompt + "\n\n### Response:";
	}
	async compareCandidates({ candidates, problemDescription, generateSearchAreasInstruction, convertCandidateToDocument, candidatesForInitialSelection, candidatesForFinalSelection, generateRankingInstruction, extractIdentifiersFromRationale, candidateIdentifierField, getSummarisableSubstringIndices }) {
		let rationale = "";
		let selectedCandidates = [];
		await this.checkEmbedderLoaded();
		let candidateDocuments = candidates.map((candidate) => convertCandidateToDocument(candidate));
		let vectorStore = await MemoryVectorStore.fromTexts(lodash.cloneDeep(candidateDocuments), candidateDocuments.map((document, index) => index), this);
		let searchAreasPromptTemplate = this.generatePromptTemplate(generateSearchAreasInstruction(problemDescription));
		if (this.DEBUG) console.log("Formatted search areas prompt: " + searchAreasPromptTemplate);
		await this.checkTokeniserLoaded();
		if (this.tokeniser.encode(searchAreasPromptTemplate).length > this.tokeniser.model_max_length) throw new Error("Search areas instruction prompt is too long for the tokeniser model");
		await this.checkGeneratorLoaded();
		let pad_token_id = this.tokeniser.pad_token_id ?? this.tokeniser.sep_token_id ?? 0;
		let eos_token_id = this.tokeniser.sep_token_id ?? 2;
		let searchAreasReplyArray = await this.generator(searchAreasPromptTemplate, {
			max_new_tokens: this.generateSearchAreasMaxNewTokens,
			temperature: this.generateSearchAreasTemperature,
			repetition_penalty: this.generateSearchAreasRepetitionPenalty,
			pad_token_id,
			eos_token_id
		});
		let searchAreasReply = Array.isArray(searchAreasReplyArray?.[0]) ? searchAreasReplyArray?.[0]?.[0] : searchAreasReplyArray?.[0];
		if (!searchAreasReply.generated_text) throw new Error("No generated text for search areas");
		if (this.DEBUG) console.log("Generated search areas response: " + searchAreasReply.generated_text);
		let searchAreasResponseIndex = searchAreasReply.generated_text.toString().indexOf("### Response:");
		if (searchAreasResponseIndex >= 0) searchAreasResponseIndex += 13;
		else searchAreasResponseIndex = 0;
		let vectorSearchQuery = searchAreasReply.generated_text.toString().substring(searchAreasResponseIndex).trim();
		if (vectorSearchQuery.includes(".")) vectorSearchQuery = vectorSearchQuery.split(".")[0].trim();
		if (this.DEBUG) console.log("Vector search query: " + vectorSearchQuery);
		let queryResult = await vectorStore.similaritySearch(vectorSearchQuery, candidatesForInitialSelection);
		await this.checkSummariserLoaded();
		let summaries = (await Promise.allSettled(queryResult.map(async (result) => {
			if (!result.pageContent || typeof result.pageContent !== "string") return "";
			if (result.pageContent.trim().split(/\s+/).length <= this.targetSummarisedStringTokenCount) return result.pageContent;
			let summarisableSubstringIndices = {
				start: 0,
				end: result.pageContent.length
			};
			if (getSummarisableSubstringIndices) Object.assign(summarisableSubstringIndices, getSummarisableSubstringIndices(result.pageContent));
			summarisableSubstringIndices.start = lodash.clamp(lodash.toSafeInteger(summarisableSubstringIndices.start), 0, result.pageContent.length);
			summarisableSubstringIndices.end = lodash.clamp(lodash.toSafeInteger(summarisableSubstringIndices.end), 0, result.pageContent.length);
			let summarisableSubstring = result.pageContent.substring(summarisableSubstringIndices.start, summarisableSubstringIndices.end);
			let contentBefore = result.pageContent.substring(0, summarisableSubstringIndices.start);
			let contentAfter = result.pageContent.substring(summarisableSubstringIndices.end);
			let wordsWithoutSummarisable = contentBefore.split(/s+/).length + contentAfter.split(/s+/).length;
			let targetSummarisedSubstringTokenCount = Math.max(1, 420 - wordsWithoutSummarisable);
			let summarisedSubstringArray = await this.summariser(summarisableSubstring, { max_length: targetSummarisedSubstringTokenCount });
			let summarisedString = contentBefore + ((Array.isArray(summarisedSubstringArray?.[0]) ? summarisedSubstringArray?.[0]?.[0] : summarisedSubstringArray?.[0])?.summary_text ?? "").split(/s+/).slice(targetSummarisedSubstringTokenCount).join(" ") + contentAfter;
			if (this.DEBUG) console.log("Summarised candidate: " + summarisedString);
		}))).filter((result) => result.status === "fulfilled" && result.value).map((result) => result.value);
		let rankingPromptTemplate = this.generatePromptTemplate(generateRankingInstruction(problemDescription, summaries));
		if (this.tokeniser.encode(rankingPromptTemplate).length > this.tokeniser.model_max_length) throw new Error("Ranking instruction prompt is too long for the tokeniser model");
		let rankingArray = await this.generator(rankingPromptTemplate, {
			max_new_tokens: this.rankingMaxNewTokens,
			temperature: this.rankingTemperature,
			repetition_penalty: this.rankingRepetitionPenalty,
			pad_token_id,
			eos_token_id
		});
		rationale = (Array.isArray(rankingArray?.[0]) ? rankingArray?.[0]?.[0] : rankingArray[0]).generated_text.toString().trim().replace(/(\*\*)|(<\/?s>)|(\[.*?\])\s*/g, "");
		if (this.DEBUG) console.log("Generated rationale: " + rationale);
		let rationaleResponseIndex = rationale.indexOf("### Response:");
		if (rationaleResponseIndex >= 0) rationaleResponseIndex += 13;
		else rationaleResponseIndex = 0;
		rationale = rationale.substring(rationaleResponseIndex);
		if (!rationale) throw new Error("No rationale generated");
		let identifiers = extractIdentifiersFromRationale(rationale);
		if (identifiers.length > candidatesForFinalSelection) identifiers = identifiers.slice(0, candidatesForFinalSelection);
		selectedCandidates = identifiers.map((identifier) => {
			let selectedCandidate = candidates.find((candidate) => String(candidate[candidateIdentifierField]).toLowerCase() === identifier.toLowerCase());
			if (selectedCandidate) return selectedCandidate;
			selectedCandidate = candidates.find((candidate) => String(candidate[candidateIdentifierField]).toLowerCase().includes(identifier.toLowerCase()));
			if (selectedCandidate) return selectedCandidate;
			selectedCandidate = candidates.find((candidate) => identifier.toLowerCase().includes(String(candidate[candidateIdentifierField]).toLowerCase()));
			if (selectedCandidate) return selectedCandidate;
			return null;
		}).filter(Boolean);
		if (this.DEBUG) console.log("Selected candidates", selectedCandidates);
		return {
			rationale,
			selectedCandidates
		};
	}
};
var src_default = AICompareCandidates;

//#endregion
export { AICompareCandidates, src_default as default };