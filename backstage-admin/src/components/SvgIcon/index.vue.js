/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { computed } from 'vue';
// 定义组件名称
defineOptions({
    name: 'svg-icon'
});
// 定义props
const props = defineProps({
    iconClass: {
        type: String,
        required: true
    },
    className: {
        type: String
    }
});
const iconName = computed(() => {
    return `#icon-${props.iconClass}`;
});
const svgClass = computed(() => {
    if (props.className) {
        return 'svg-icon ' + props.className;
    }
    else {
        return 'svg-icon';
    }
});
const __VLS_ctx = {
    ...{},
    ...{},
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
__VLS_asFunctionalElement1(__VLS_intrinsics.svg, __VLS_intrinsics.svg)({
    ...{ class: (__VLS_ctx.svgClass) },
    'aria-hidden': "true",
});
__VLS_asFunctionalElement1(__VLS_intrinsics.use, __VLS_intrinsics.use)({
    'xlink:href': (__VLS_ctx.iconName),
});
// @ts-ignore
[svgClass, iconName,];
const __VLS_export = (await import('vue')).defineComponent({
    props: {
        iconClass: {
            type: String,
            required: true
        },
        className: {
            type: String
        }
    },
});
export default {};
