/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { computed, ref, watch } from 'vue';
import { useRoute } from 'vue-router';
import { ElBreadcrumb, ElBreadcrumbItem } from 'element-plus';
// 定义组件名称
defineOptions({
    name: 'Breadcrumb'
});
// 显式注册Element Plus组件
const __VLS_exposed = {
    ElBreadcrumb,
    ElBreadcrumbItem
};
defineExpose(__VLS_exposed);
// 响应式数据
const levelList = ref([]);
const route = useRoute();
// 计算属性，只返回有标题的面包屑项
const visibleLevelList = computed(() => {
    return levelList.value.filter(item => item.meta?.title);
});
// 获取面包屑路径
const getBreadcrumb = () => {
    let matched = route.matched.filter(item => item.name);
    const first = matched[0];
    if (first && first.name !== 'home') {
        const homeRoute = { ...first, path: '/home', meta: { title: '首页' }, name: 'home' };
        matched = [homeRoute].concat(matched);
    }
    levelList.value = matched;
};
// 监听路由变化
watch(route, () => {
    getBreadcrumb();
}, { immediate: true });
// 初始化
getBreadcrumb();
const __VLS_ctx = {
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elBreadcrumb | typeof __VLS_components.ElBreadcrumb | typeof __VLS_components['el-breadcrumb'] | typeof __VLS_components.elBreadcrumb | typeof __VLS_components.ElBreadcrumb | typeof __VLS_components['el-breadcrumb']} */
elBreadcrumb;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    ...{ class: "app-breadcrumb" },
    separator: "/",
}));
const __VLS_2 = __VLS_1({
    ...{ class: "app-breadcrumb" },
    separator: "/",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_5 = {};
/** @type {__VLS_StyleScopedClasses['app-breadcrumb']} */ ;
const { default: __VLS_6 } = __VLS_3.slots;
let __VLS_7;
/** @ts-ignore @type { | typeof __VLS_components.transitionGroup | typeof __VLS_components.TransitionGroup | typeof __VLS_components['transition-group'] | typeof __VLS_components.transitionGroup | typeof __VLS_components.TransitionGroup | typeof __VLS_components['transition-group']} */
transitionGroup;
// @ts-ignore
const __VLS_8 = __VLS_asFunctionalComponent1(__VLS_7, new __VLS_7({
    name: "breadcrumb",
}));
const __VLS_9 = __VLS_8({
    name: "breadcrumb",
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
const { default: __VLS_12 } = __VLS_10.slots;
for (const [item, index] of __VLS_vFor((__VLS_ctx.visibleLevelList))) {
    let __VLS_13;
    /** @ts-ignore @type { | typeof __VLS_components.elBreadcrumbItem | typeof __VLS_components.ElBreadcrumbItem | typeof __VLS_components['el-breadcrumb-item'] | typeof __VLS_components.elBreadcrumbItem | typeof __VLS_components.ElBreadcrumbItem | typeof __VLS_components['el-breadcrumb-item']} */
    elBreadcrumbItem;
    // @ts-ignore
    const __VLS_14 = __VLS_asFunctionalComponent1(__VLS_13, new __VLS_13({
        key: (item.path),
    }));
    const __VLS_15 = __VLS_14({
        key: (item.path),
    }, ...__VLS_functionalComponentArgsRest(__VLS_14));
    const { default: __VLS_18 } = __VLS_16.slots;
    if (item.redirect === 'noredirect' || index == __VLS_ctx.visibleLevelList.length - 1) {
        __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
            ...{ class: "no-redirect" },
        });
        /** @type {__VLS_StyleScopedClasses['no-redirect']} */ ;
        (item.meta.title);
    }
    else {
        let __VLS_19;
        /** @ts-ignore @type { | typeof __VLS_components.routerLink | typeof __VLS_components.RouterLink | typeof __VLS_components['router-link'] | typeof __VLS_components.routerLink | typeof __VLS_components.RouterLink | typeof __VLS_components['router-link']} */
        routerLink;
        // @ts-ignore
        const __VLS_20 = __VLS_asFunctionalComponent1(__VLS_19, new __VLS_19({
            to: (typeof item.redirect === 'string' ? item.redirect : item.path),
        }));
        const __VLS_21 = __VLS_20({
            to: (typeof item.redirect === 'string' ? item.redirect : item.path),
        }, ...__VLS_functionalComponentArgsRest(__VLS_20));
        const { default: __VLS_24 } = __VLS_22.slots;
        (item.meta.title);
        // @ts-ignore
        [visibleLevelList, visibleLevelList,];
        var __VLS_22;
    }
    // @ts-ignore
    [];
    var __VLS_16;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_10;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({
    setup: () => __VLS_exposed,
});
export default {};
