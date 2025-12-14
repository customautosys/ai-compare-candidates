import {defineBoot} from '#q-app/wrappers';
import * as fontawesomeImport from '@quasar/extras/fontawesome-v6';
import * as materialIconsImport from '@quasar/extras/material-icons';
import type {QVueGlobals} from 'quasar';

export default defineBoot(({app})=>{
	(app.config.globalProperties.$q as QVueGlobals).iconMapFn=iconName=>{
		const fontawesome=fontawesomeImport as {[key:string]:string};
		const materialIcons=materialIconsImport as {[key:string]:string};
		if(materialIcons[iconName])return {icon:materialIcons[iconName]};
		if(fontawesome[iconName])return {icon:fontawesome[iconName]};
		let matches=iconName.match(/(fa[A-Za-z]) fa-(.*?)(as {[key:string]:s\s|$)/);
		if(matches&&matches[1]&&matches[2]){
			let icon=fontawesome[matches[1]+matches[2].replace(/(^|-)([a-z])/g,letters=>String(letters[letters.length-1]).toUpperCase())];
			if(icon)return {icon};
		}
		let icon=materialIcons['mat'+iconName.replace(/(^|-)([a-z])/g,letters=>String(letters[letters.length-1]).toUpperCase())];
		if(icon)return {icon};
	}
});