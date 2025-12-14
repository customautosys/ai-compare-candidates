import {defineBoot} from '#q-app/wrappers';

export default defineBoot(()=>{
	if(process.env.MODE==='electron'){
		delete (process.release as any).name;
		delete (process.versions as any).node;
	}
});