/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import SidebarItem from './SidebarItem.vue';
import ScrollBar from '@/components/ScrollBar/index.vue';
import { useAppStore } from '@/stores/app';
import usePermissionStore from '@/stores/permission';
// 定义组件名称
defineOptions({
    name: 'Sidebar'
});
// 使用 Pinia stores
const appStore = useAppStore();
const permissionStore = usePermissionStore();
const route = useRoute();
// 所有路由
const routes = computed(() => permissionStore.routers);
// 侧边栏打开状态
const isCollapse = computed(() => !appStore.sidebar.opened);
const __VLS_ctx = {
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
const __VLS_0 = ScrollBar || ScrollBar;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({}));
const __VLS_2 = __VLS_1({}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_5 = {};
const { default: __VLS_6 } = __VLS_3.slots;
let __VLS_7;
/** @ts-ignore @type { | typeof __VLS_components.elMenu | typeof __VLS_components.ElMenu | typeof __VLS_components['el-menu'] | typeof __VLS_components.elMenu | typeof __VLS_components.ElMenu | typeof __VLS_components['el-menu']} */
elMenu;
// @ts-ignore
const __VLS_8 = __VLS_asFunctionalComponent1(__VLS_7, new __VLS_7({
    mode: "vertical",
    showTimeout: (200),
    defaultActive: (__VLS_ctx.route.path),
    collapse: (__VLS_ctx.isCollapse),
    backgroundColor: "#304156",
    textColor: "#bfcbd9",
    activeTextColor: "#409EFF",
}));
const __VLS_9 = __VLS_8({
    mode: "vertical",
    showTimeout: (200),
    defaultActive: (__VLS_ctx.route.path),
    collapse: (__VLS_ctx.isCollapse),
    backgroundColor: "#304156",
    textColor: "#bfcbd9",
    activeTextColor: "#409EFF",
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
const { default: __VLS_12 } = __VLS_10.slots;
const __VLS_13 = SidebarItem || SidebarItem;
// @ts-ignore
const __VLS_14 = __VLS_asFunctionalComponent1(__VLS_13, new __VLS_13({
    routes: (__VLS_ctx.routes),
}));
const __VLS_15 = __VLS_14({
    routes: (__VLS_ctx.routes),
}, ...__VLS_functionalComponentArgsRest(__VLS_14));
// @ts-ignore
[route, isCollapse, routes,];
var __VLS_10;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
