import { pipeline } from "@huggingface/transformers";

//#region src/index.ts
var AICompareCandidates = class {
	constructor() {
		this.generatorModelName = "Xenova/LaMini-GPT-774m";
		this.generatorPromise = null;
		this.generatorProgressInfo = null;
		this.generatorProgressCallback = null;
	}
	async loadGenerator(generatorProgressCallback, generatorModelName = "") {
		if (typeof generatorModelName === "string" && generatorModelName) this.generatorModelName = generatorModelName;
		if (generatorProgressCallback) this.generatorProgressCallback = generatorProgressCallback;
		this.generatorPromise = pipeline("text-generation", this.generatorModelName, {
			device: "webgpu",
			progress_callback: (progressInfo) => {
				this.generatorProgressInfo = progressInfo;
				return this.generatorProgressCallback(progressInfo);
			}
		});
	}
};
var src_default = AICompareCandidates;

//#endregion
export { AICompareCandidates, src_default as default };