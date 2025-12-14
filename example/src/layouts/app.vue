<template>
	<q-layout view="lHh Lpr lFf">
		<q-header elevated>
			<q-toolbar>
				<q-toolbar-title>
					AI Compare Candidates Example
				</q-toolbar-title>
			</q-toolbar>
		</q-header>
		<q-page-container>
			<q-page>
				<q-virtual-scroll :items="candidates">
					<template #default="{item,index}:{item:Candidate,index:number}">
						<q-item>
							<q-item-section side>
								<q-item-label>{{index+1}}.</q-item-label>
							</q-item-section>
							<q-item-section>
								<q-input label="Name" v-model="item.name"/>
							</q-item-section>
							<q-item-section side>
								<q-btn no-caps icon="fas fa-trash-can" label="Remove Candidate" @click="removeCandidate(index)"/>
							</q-item-section>
						</q-item>
					</template>
				</q-virtual-scroll>
				<q-list>
					<q-item>
						<q-item-section>
							<q-input autogrow label="Artificial Intelligence Problem Description Prompt" v-model="problemDescription"/>
						</q-item-section>
					</q-item>
					<q-item>
						<q-item-section>
							<q-btn no-caps icon="fas fa-plus" label="Add Candidate" @click="addCandidate"/>
						</q-item-section>
						<q-item-section>
							<q-btn no-caps icon="fas fa-brain" label="Artificial Intelligence" @click="artificialIntelligence"/>
						</q-item-section>
					</q-item>
					<q-item>
						<q-item-section>
							<q-item-label>Rationale:</q-item-label>
							<q-item-label>{{outcome.rationale}}</q-item-label>
						</q-item-section>
					</q-item>
				</q-list>
				<q-virtual-scroll :items="outcome.selectedCandidates">
					<template #before>
						<q-item>
							<q-item-section>
								<q-item-label>Selected Candidates:</q-item-label>
							</q-item-section>
						</q-item>
					</template>
					<template #default="{item,index}:{item:Candidate,index:number}">
						<q-item>
							<q-item-section side>
								<q-item-label>{{index+1}}.</q-item-label>
							</q-item-section>
							<q-item-section>
								<q-item-label>{{item.name}}</q-item-label>
							</q-item-section>
						</q-item>
					</template>
				</q-virtual-scroll>
			</q-page>
		</q-page-container>
	</q-layout>
</template>

<script setup lang="ts">
import {ref} from 'vue';
import{
	Dialog,
	Loading
}from 'quasar';
import jsan from 'jsan';
import useStore from '../stores/store';
import type AICompareCandidates from 'ai-compare-candidates';

interface Candidate{
	name:string;
}

const store=useStore();

const candidates=ref<Candidate[]>([{name:'Testing 1'},{name:'Testing 2'}]);
const problemDescription=ref('Pick the best name');
const outcome=ref<AICompareCandidates.CompareCandidatesReturn<Candidate>>({
	selectedCandidates:[],
	rationale:''
});

function removeCandidate(index:number){
	if(candidates.value.length<=2){
		Dialog.create({message:'There must be at least 2 candidates to compare'});
		return;
	}
	candidates.value.splice(index,1);
}

function addCandidate(){
	candidates.value.push({name:''});
}

function errorMessage(error:any){
	return typeof error?.response?.data==='string'?error.response.data:error?.response?.data?jsan.stringify(error.response.data):typeof error?.message==='string'?error.message:error?.message?jsan.stringify(error.message):typeof error==='string'?error:jsan.stringify(error);
}

async function artificialIntelligence(){
	try{
		Loading.show();
		outcome.value=await store.aiCompareCandidates.compareCandidates({
			candidates:candidates.value,
			problemDescription:problemDescription.value
		});
	}catch(error){
		console.log(error);
		Dialog.create({message:errorMessage(error)});
	}finally{
		Loading.hide();
	}
}
</script>