import{
	defineStore,
	acceptHMRUpdate
}from 'pinia';
import {shallowRef} from 'vue';
import AICompareCandidates from 'ai-compare-candidates';

export const useStore=defineStore('store',()=>{
	const aiCompareCandidates=shallowRef(new AICompareCandidates());
	return{
		aiCompareCandidates
	};
});

export default useStore;

if(import.meta.hot){
	import.meta.hot.accept(acceptHMRUpdate(useStore, import.meta.hot));
}
