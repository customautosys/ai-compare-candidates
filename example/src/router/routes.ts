import type {RouteRecordRaw} from 'vue-router';

const routes:RouteRecordRaw[]=[{
	path:'/:catchAll(.*)*',
	component:()=>import('layouts/app.vue')
}];

export default routes;
