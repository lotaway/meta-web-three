/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { onMounted, inject, computed, ref } from 'vue';
import { getSubjectListAllAPI } from '@/apis/subject';
import { getPrefrenceAreaListAllAPI } from '@/apis/prefrenceArea';
// 定义属性
const props = defineProps({
    isEdit: {
        type: Boolean,
        default: false
    }
});
// 定义事件
const emit = defineEmits(['prev-step', 'finish-commit']);
// 获取跨层传递的数据
const compProductParam = inject('product-key');
// 所有专题列表
const subjectList = ref([]);
// 所有优选专区列表
const prefrenceAreaList = ref([]);
// 获取专题列表
const getSubjectList = async () => {
    const res = await getSubjectListAllAPI();
    subjectList.value = res.data.map(item => ({
        label: item.title,
        key: item.id
    }));
};
// 获取优选区域列表
const getPrefrenceAreaList = async () => {
    const res = await getPrefrenceAreaListAllAPI();
    prefrenceAreaList.value = res.data.map(item => ({
        label: item.name,
        key: item.id
    }));
};
// 初始化数据
onMounted(() => {
    getSubjectList();
    getPrefrenceAreaList();
});
// 关联专题
const selectSubject = computed({
    get: function () {
        return compProductParam.value.subjectProductRelationList?.map(item => item.subjectId);
    },
    set: function (newValue) {
        compProductParam.value.subjectProductRelationList = [];
        newValue?.forEach(item => {
            compProductParam.value.subjectProductRelationList.push({ subjectId: item });
        });
    }
});
// 关联优选
const selectPrefrenceArea = computed({
    get: function () {
        return compProductParam.value.prefrenceAreaProductRelationList?.map(item => item.prefrenceAreaId);
    },
    set: function (newValue) {
        compProductParam.value.prefrenceAreaProductRelationList = [];
        newValue?.forEach(item => {
            compProductParam.value.prefrenceAreaProductRelationList.push({ prefrenceAreaId: item });
        });
    }
});
// 过滤方法
const filterMethod = (query, item) => {
    return item.label.indexOf(query) > -1;
};
// 上一步处理
const handlePrev = () => {
    emit('prev-step');
};
// 完成提交处理
const handleFinishCommit = async () => {
    emit('finish-commit', props.isEdit);
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
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    model: (__VLS_ctx.compProductParam),
    ref: "productRelationForm",
    labelWidth: "120px",
    ...{ class: "form-inner-container" },
}));
const __VLS_2 = __VLS_1({
    model: (__VLS_ctx.compProductParam),
    ref: "productRelationForm",
    labelWidth: "120px",
    ...{ class: "form-inner-container" },
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_5 = {};
/** @type {__VLS_StyleScopedClasses['form-inner-container']} */ ;
const { default: __VLS_7 } = __VLS_3.slots;
let __VLS_8;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_9 = __VLS_asFunctionalComponent1(__VLS_8, new __VLS_8({
    label: "关联专题：",
}));
const __VLS_10 = __VLS_9({
    label: "关联专题：",
}, ...__VLS_functionalComponentArgsRest(__VLS_9));
const { default: __VLS_13 } = __VLS_11.slots;
let __VLS_14;
/** @ts-ignore @type { | typeof __VLS_components.elTransfer | typeof __VLS_components.ElTransfer | typeof __VLS_components['el-transfer'] | typeof __VLS_components.elTransfer | typeof __VLS_components.ElTransfer | typeof __VLS_components['el-transfer']} */
elTransfer;
// @ts-ignore
const __VLS_15 = __VLS_asFunctionalComponent1(__VLS_14, new __VLS_14({
    ...{ style: {} },
    filterable: true,
    filterMethod: (__VLS_ctx.filterMethod),
    filterPlaceholder: "请输入专题名称",
    modelValue: (__VLS_ctx.selectSubject),
    titles: (['待选择', '已选择']),
    data: (__VLS_ctx.subjectList),
}));
const __VLS_16 = __VLS_15({
    ...{ style: {} },
    filterable: true,
    filterMethod: (__VLS_ctx.filterMethod),
    filterPlaceholder: "请输入专题名称",
    modelValue: (__VLS_ctx.selectSubject),
    titles: (['待选择', '已选择']),
    data: (__VLS_ctx.subjectList),
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
// @ts-ignore
[compProductParam, filterMethod, selectSubject, subjectList,];
var __VLS_11;
let __VLS_19;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_20 = __VLS_asFunctionalComponent1(__VLS_19, new __VLS_19({
    label: "关联优选：",
}));
const __VLS_21 = __VLS_20({
    label: "关联优选：",
}, ...__VLS_functionalComponentArgsRest(__VLS_20));
const { default: __VLS_24 } = __VLS_22.slots;
let __VLS_25;
/** @ts-ignore @type { | typeof __VLS_components.elTransfer | typeof __VLS_components.ElTransfer | typeof __VLS_components['el-transfer'] | typeof __VLS_components.elTransfer | typeof __VLS_components.ElTransfer | typeof __VLS_components['el-transfer']} */
elTransfer;
// @ts-ignore
const __VLS_26 = __VLS_asFunctionalComponent1(__VLS_25, new __VLS_25({
    ...{ style: {} },
    filterable: true,
    filterMethod: (__VLS_ctx.filterMethod),
    filterPlaceholder: "请输入优选名称",
    modelValue: (__VLS_ctx.selectPrefrenceArea),
    titles: (['待选择', '已选择']),
    data: (__VLS_ctx.prefrenceAreaList),
}));
const __VLS_27 = __VLS_26({
    ...{ style: {} },
    filterable: true,
    filterMethod: (__VLS_ctx.filterMethod),
    filterPlaceholder: "请输入优选名称",
    modelValue: (__VLS_ctx.selectPrefrenceArea),
    titles: (['待选择', '已选择']),
    data: (__VLS_ctx.prefrenceAreaList),
}, ...__VLS_functionalComponentArgsRest(__VLS_26));
// @ts-ignore
[filterMethod, selectPrefrenceArea, prefrenceAreaList,];
var __VLS_22;
let __VLS_30;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_31 = __VLS_asFunctionalComponent1(__VLS_30, new __VLS_30({}));
const __VLS_32 = __VLS_31({}, ...__VLS_functionalComponentArgsRest(__VLS_31));
const { default: __VLS_35 } = __VLS_33.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_36;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_37 = __VLS_asFunctionalComponent1(__VLS_36, new __VLS_36({
    ...{ 'onClick': {} },
}));
const __VLS_38 = __VLS_37({
    ...{ 'onClick': {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_37));
let __VLS_41;
const __VLS_42 = ({ click: {} },
    { onClick: (__VLS_ctx.handlePrev) });
const { default: __VLS_43 } = __VLS_39.slots;
// @ts-ignore
[handlePrev,];
var __VLS_39;
var __VLS_40;
let __VLS_44;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_45 = __VLS_asFunctionalComponent1(__VLS_44, new __VLS_44({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_46 = __VLS_45({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_45));
let __VLS_49;
const __VLS_50 = ({ click: {} },
    { onClick: (__VLS_ctx.handleFinishCommit) });
const { default: __VLS_51 } = __VLS_47.slots;
// @ts-ignore
[handleFinishCommit,];
var __VLS_47;
var __VLS_48;
// @ts-ignore
[];
var __VLS_33;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
var __VLS_6 = __VLS_5;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({
    emits: {},
    props: {
        isEdit: {
            type: Boolean,
            default: false
        }
    },
});
export default {};
