/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { computed } from 'vue';
import { useAppStore } from '@/stores/app';
import Navbar from './components/Navbar.vue';
import Sidebar from './components/Sidebar/index.vue';
import AppMain from './components/AppMain.vue';
import useResizeHandler from './composables/useResizeHandler';
// 使用 Pinia store
const appStore = useAppStore();
// 获取响应式状态
const sidebar = computed(() => appStore.sidebar);
const device = computed(() => appStore.device);
// 计算类名
const classObj = computed(() => ({
    hideSidebar: !sidebar.value.opened,
    withoutAnimation: sidebar.value.withoutAnimation,
    mobile: device.value === 'mobile'
}));
// 使用 resize handler composable
useResizeHandler();
const __VLS_ctx = {
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "app-wrapper" },
    ...{ class: (__VLS_ctx.classObj) },
});
/** @type {__VLS_StyleScopedClasses['app-wrapper']} */ ;
const __VLS_0 = Sidebar || Sidebar;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    ...{ class: "sidebar-container" },
}));
const __VLS_2 = __VLS_1({
    ...{ class: "sidebar-container" },
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
/** @type {__VLS_StyleScopedClasses['sidebar-container']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "main-container" },
});
/** @type {__VLS_StyleScopedClasses['main-container']} */ ;
const __VLS_5 = Navbar || Navbar;
// @ts-ignore
const __VLS_6 = __VLS_asFunctionalComponent1(__VLS_5, new __VLS_5({}));
const __VLS_7 = __VLS_6({}, ...__VLS_functionalComponentArgsRest(__VLS_6));
const __VLS_10 = AppMain || AppMain;
// @ts-ignore
const __VLS_11 = __VLS_asFunctionalComponent1(__VLS_10, new __VLS_10({}));
const __VLS_12 = __VLS_11({}, ...__VLS_functionalComponentArgsRest(__VLS_11));
// @ts-ignore
[classObj,];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
