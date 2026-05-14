/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, computed } from 'vue';
// 定义组件的props
const props = defineProps({
    modelValue: Boolean
});
// 定义组件的emits
const emit = defineEmits(['update:modelValue']);
// 默认物流列表数据
const defaultLogisticsList = [
    { name: '订单已提交，等待付款', time: '2017-04-01 12:00:00 ' },
    { name: '订单付款成功', time: '2017-04-01 12:00:00 ' },
    { name: '在北京市进行下级地点扫描，等待付款', time: '2017-04-01 12:00:00 ' },
    { name: '在分拨中心广东深圳公司进行卸车扫描，等待付款', time: '2017-04-01 12:00:00 ' },
    { name: '在广东深圳公司进行发出扫描', time: '2017-04-01 12:00:00 ' },
    { name: '到达目的地网点广东深圳公司，快件将很快进行派送', time: '2017-04-01 12:00:00 ' },
    { name: '订单已签收，期待再次为您服务', time: '2017-04-01 12:00:00 ' }
];
// 物流列表数据
const logisticsList = ref(Object.assign([], defaultLogisticsList));
// 控制对话框显示的计算属性
const visible = computed({
    get() {
        return props.modelValue;
    },
    set(visible) {
        emit('update:modelValue', visible);
    }
});
// 发出输入事件
const emitInput = (val) => {
    emit('update:modelValue', val);
};
// 处理关闭对话框
const handleClose = () => {
    emitInput(false);
};
const __VLS_ctx = {
    ...{},
    ...{},
    ...{},
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    title: "订单跟踪",
    modelValue: (__VLS_ctx.visible),
    beforeClose: (__VLS_ctx.handleClose),
    width: "40%",
}));
const __VLS_2 = __VLS_1({
    title: "订单跟踪",
    modelValue: (__VLS_ctx.visible),
    beforeClose: (__VLS_ctx.handleClose),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_5 = {};
const { default: __VLS_6 } = __VLS_3.slots;
let __VLS_7;
/** @ts-ignore @type { | typeof __VLS_components.elSteps | typeof __VLS_components.ElSteps | typeof __VLS_components['el-steps'] | typeof __VLS_components.elSteps | typeof __VLS_components.ElSteps | typeof __VLS_components['el-steps']} */
elSteps;
// @ts-ignore
const __VLS_8 = __VLS_asFunctionalComponent1(__VLS_7, new __VLS_7({
    direction: "vertical",
    active: (6),
    finishStatus: "success",
    space: "50px",
}));
const __VLS_9 = __VLS_8({
    direction: "vertical",
    active: (6),
    finishStatus: "success",
    space: "50px",
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
const { default: __VLS_12 } = __VLS_10.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.logisticsList))) {
    let __VLS_13;
    /** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
    elStep;
    // @ts-ignore
    const __VLS_14 = __VLS_asFunctionalComponent1(__VLS_13, new __VLS_13({
        key: (item.name),
        title: (item.name),
        description: (item.time),
    }));
    const __VLS_15 = __VLS_14({
        key: (item.name),
        title: (item.name),
        description: (item.time),
    }, ...__VLS_functionalComponentArgsRest(__VLS_14));
    // @ts-ignore
    [visible, handleClose, logisticsList,];
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
    emits: {},
    props: {
        modelValue: Boolean
    },
});
export default {};
