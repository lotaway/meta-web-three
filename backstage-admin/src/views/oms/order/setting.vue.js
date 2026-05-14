/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getOrderSettingByIdAPI, orderSettingUpdateByIdAPI } from '@/apis/orderSetting';
// 默认订单设置数据
const defaultOrderSetting = {
    id: 1,
    flashOrderOvertime: 0,
    normalOrderOvertime: 0,
    confirmOvertime: 0,
    finishOvertime: 0,
    commentOvertime: 0
};
// 订单设置数据
const orderSetting = ref(Object.assign({}, defaultOrderSetting));
// 获取详情
const getDetail = async () => {
    const response = await getOrderSettingByIdAPI(orderSetting.value.id);
    orderSetting.value = response.data;
};
// 组件挂载后获取详情
onMounted(() => {
    getDetail();
});
// 订单设置表单引用
const orderSettingForm = ref();
// 时间验证规则
const checkTime = (rule, value, callback) => {
    if (!value) {
        return callback(new Error('时间不能为空'));
    }
    const intValue = parseInt(value);
    if (!Number.isInteger(intValue)) {
        return callback(new Error('请输入数字值'));
    }
    callback();
};
// 表单验证规则
const rules = ref({
    flashOrderOvertime: { validator: checkTime, trigger: 'blur' },
    normalOrderOvertime: { validator: checkTime, trigger: 'blur' },
    confirmOvertime: { validator: checkTime, trigger: 'blur' },
    finishOvertime: { validator: checkTime, trigger: 'blur' },
    commentOvertime: { validator: checkTime, trigger: 'blur' }
});
// 确认提交表单
const confirm = async () => {
    if (!orderSettingForm.value)
        return;
    const valid = await orderSettingForm.value.validate();
    if (valid) {
        await ElMessageBox.confirm('是否要提交修改?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await orderSettingUpdateByIdAPI(1, orderSetting.value);
        ElMessage({
            type: 'success',
            message: '提交成功!',
            duration: 1000
        });
    }
    else {
        ElMessage({
            message: '提交参数不合法',
            type: 'warning'
        });
        return false;
    }
};
const __VLS_ctx = {
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    ...{ class: "form-container" },
    shadow: "never",
}));
const __VLS_2 = __VLS_1({
    ...{ class: "form-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_5 = {};
/** @type {__VLS_StyleScopedClasses['form-container']} */ ;
const { default: __VLS_6 } = __VLS_3.slots;
let __VLS_7;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_8 = __VLS_asFunctionalComponent1(__VLS_7, new __VLS_7({
    model: (__VLS_ctx.orderSetting),
    ref: "orderSettingForm",
    rules: (__VLS_ctx.rules),
    labelWidth: "150px",
}));
const __VLS_9 = __VLS_8({
    model: (__VLS_ctx.orderSetting),
    ref: "orderSettingForm",
    rules: (__VLS_ctx.rules),
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
var __VLS_12 = {};
const { default: __VLS_14 } = __VLS_10.slots;
let __VLS_15;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_16 = __VLS_asFunctionalComponent1(__VLS_15, new __VLS_15({
    label: "秒杀订单超过：",
    prop: "flashOrderOvertime",
}));
const __VLS_17 = __VLS_16({
    label: "秒杀订单超过：",
    prop: "flashOrderOvertime",
}, ...__VLS_functionalComponentArgsRest(__VLS_16));
const { default: __VLS_20 } = __VLS_18.slots;
let __VLS_21;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_22 = __VLS_asFunctionalComponent1(__VLS_21, new __VLS_21({
    modelValue: (__VLS_ctx.orderSetting.flashOrderOvertime),
    ...{ class: "input-width" },
}));
const __VLS_23 = __VLS_22({
    modelValue: (__VLS_ctx.orderSetting.flashOrderOvertime),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_22));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_26 } = __VLS_24.slots;
{
    const { append: __VLS_27 } = __VLS_24.slots;
    // @ts-ignore
    [orderSetting, orderSetting, rules,];
}
// @ts-ignore
[];
var __VLS_24;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "note-margin" },
});
/** @type {__VLS_StyleScopedClasses['note-margin']} */ ;
// @ts-ignore
[];
var __VLS_18;
let __VLS_28;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_29 = __VLS_asFunctionalComponent1(__VLS_28, new __VLS_28({
    label: "正常订单超过：",
    prop: "normalOrderOvertime",
}));
const __VLS_30 = __VLS_29({
    label: "正常订单超过：",
    prop: "normalOrderOvertime",
}, ...__VLS_functionalComponentArgsRest(__VLS_29));
const { default: __VLS_33 } = __VLS_31.slots;
let __VLS_34;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_35 = __VLS_asFunctionalComponent1(__VLS_34, new __VLS_34({
    modelValue: (__VLS_ctx.orderSetting.normalOrderOvertime),
    ...{ class: "input-width" },
}));
const __VLS_36 = __VLS_35({
    modelValue: (__VLS_ctx.orderSetting.normalOrderOvertime),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_35));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_39 } = __VLS_37.slots;
{
    const { append: __VLS_40 } = __VLS_37.slots;
    // @ts-ignore
    [orderSetting,];
}
// @ts-ignore
[];
var __VLS_37;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "note-margin" },
});
/** @type {__VLS_StyleScopedClasses['note-margin']} */ ;
// @ts-ignore
[];
var __VLS_31;
let __VLS_41;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_42 = __VLS_asFunctionalComponent1(__VLS_41, new __VLS_41({
    label: "发货超过：",
    prop: "confirmOvertime",
}));
const __VLS_43 = __VLS_42({
    label: "发货超过：",
    prop: "confirmOvertime",
}, ...__VLS_functionalComponentArgsRest(__VLS_42));
const { default: __VLS_46 } = __VLS_44.slots;
let __VLS_47;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({
    modelValue: (__VLS_ctx.orderSetting.confirmOvertime),
    ...{ class: "input-width" },
}));
const __VLS_49 = __VLS_48({
    modelValue: (__VLS_ctx.orderSetting.confirmOvertime),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_48));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_52 } = __VLS_50.slots;
{
    const { append: __VLS_53 } = __VLS_50.slots;
    // @ts-ignore
    [orderSetting,];
}
// @ts-ignore
[];
var __VLS_50;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "note-margin" },
});
/** @type {__VLS_StyleScopedClasses['note-margin']} */ ;
// @ts-ignore
[];
var __VLS_44;
let __VLS_54;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    label: "订单完成超过：",
    prop: "finishOvertime",
}));
const __VLS_56 = __VLS_55({
    label: "订单完成超过：",
    prop: "finishOvertime",
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
const { default: __VLS_59 } = __VLS_57.slots;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    modelValue: (__VLS_ctx.orderSetting.finishOvertime),
    ...{ class: "input-width" },
}));
const __VLS_62 = __VLS_61({
    modelValue: (__VLS_ctx.orderSetting.finishOvertime),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_65 } = __VLS_63.slots;
{
    const { append: __VLS_66 } = __VLS_63.slots;
    // @ts-ignore
    [orderSetting,];
}
// @ts-ignore
[];
var __VLS_63;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "note-margin" },
});
/** @type {__VLS_StyleScopedClasses['note-margin']} */ ;
// @ts-ignore
[];
var __VLS_57;
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    label: "订单完成超过：",
    prop: "commentOvertime",
}));
const __VLS_69 = __VLS_68({
    label: "订单完成超过：",
    prop: "commentOvertime",
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
const { default: __VLS_72 } = __VLS_70.slots;
let __VLS_73;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_74 = __VLS_asFunctionalComponent1(__VLS_73, new __VLS_73({
    modelValue: (__VLS_ctx.orderSetting.commentOvertime),
    ...{ class: "input-width" },
}));
const __VLS_75 = __VLS_74({
    modelValue: (__VLS_ctx.orderSetting.commentOvertime),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_74));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_78 } = __VLS_76.slots;
{
    const { append: __VLS_79 } = __VLS_76.slots;
    // @ts-ignore
    [orderSetting,];
}
// @ts-ignore
[];
var __VLS_76;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "note-margin" },
});
/** @type {__VLS_StyleScopedClasses['note-margin']} */ ;
// @ts-ignore
[];
var __VLS_70;
let __VLS_80;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_81 = __VLS_asFunctionalComponent1(__VLS_80, new __VLS_80({}));
const __VLS_82 = __VLS_81({}, ...__VLS_functionalComponentArgsRest(__VLS_81));
const { default: __VLS_85 } = __VLS_83.slots;
let __VLS_86;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_87 = __VLS_asFunctionalComponent1(__VLS_86, new __VLS_86({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_88 = __VLS_87({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_87));
let __VLS_91;
const __VLS_92 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.confirm();
            // @ts-ignore
            [confirm,];
        } });
const { default: __VLS_93 } = __VLS_89.slots;
// @ts-ignore
[];
var __VLS_89;
var __VLS_90;
// @ts-ignore
[];
var __VLS_83;
// @ts-ignore
[];
var __VLS_10;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
var __VLS_13 = __VLS_12;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
