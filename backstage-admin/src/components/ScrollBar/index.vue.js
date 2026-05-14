/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref } from 'vue';
// 定义组件名称
defineOptions({
    name: 'ScrollBar'
});
const delta = 15;
// 响应式数据
const top = ref(0);
const scrollContainer = ref();
const scrollWrapper = ref();
// 处理滚动事件
const handleScroll = (e) => {
    const eventDelta = -e.deltaY * 3;
    const $container = scrollContainer.value;
    if (!$container)
        return;
    const $containerHeight = $container.offsetHeight;
    const $wrapper = scrollWrapper.value;
    if (!$wrapper)
        return;
    const $wrapperHeight = $wrapper.offsetHeight;
    if (eventDelta > 0) {
        top.value = Math.min(0, top.value + eventDelta);
    }
    else {
        if ($containerHeight - delta < $wrapperHeight) {
            if (top.value < -($wrapperHeight - $containerHeight + delta)) {
                // 当已经到达边界时，无需重新赋值
            }
            else {
                top.value = Math.max(top.value + eventDelta, $containerHeight - $wrapperHeight - delta);
            }
        }
        else {
            top.value = 0;
        }
    }
};
const __VLS_ctx = {
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ onWheel: (__VLS_ctx.handleScroll) },
    ...{ class: "scroll-container" },
    ref: "scrollContainer",
});
/** @type {__VLS_StyleScopedClasses['scroll-container']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "scroll-wrapper" },
    ref: "scrollWrapper",
    ...{ style: ({ top: __VLS_ctx.top + 'px' }) },
});
/** @type {__VLS_StyleScopedClasses['scroll-wrapper']} */ ;
var __VLS_0 = {};
// @ts-ignore
var __VLS_1 = __VLS_0;
// @ts-ignore
[handleScroll, top,];
const __VLS_base = (await import('vue')).defineComponent({});
const __VLS_export = {};
export default {};
