/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, reactive, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import SingleUpload from '@/components/Upload/singleUpload.vue';
import { homeAdvertiseCreateAPI, getHomeAdvertiseByIdAPI, homeAdvertiseUpdateAPI } from '@/apis/homeAdvertise';
// 获取路由对象
const router = useRouter();
const route = useRoute();
// 定义属性
const props = defineProps({
    // 是否为编辑模式
    isEdit: {
        type: Boolean,
        default: false
    }
});
// 默认轮播广告，添加时使用
const defaultHomeAdvertise = {
    name: '',
    type: 1,
    status: 0,
    sort: 0
};
// 定义轮播广告对象
const homeAdvertise = ref(Object.assign({}, defaultHomeAdvertise));
// 轮播广告位置
const typeOptions = ref([
    {
        label: 'PC首页轮播',
        value: 0
    },
    {
        label: 'APP首页轮播',
        value: 1
    }
]);
// 轮播广告表单组件引用
const homeAdvertiseFrom = ref();
// 轮播广告表单校验规则
const rules = reactive({
    name: [
        { required: true, message: '请输入广告名称', trigger: 'blur' },
        { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
    ],
    url: [
        { required: true, message: '请输入广告链接', trigger: 'blur' }
    ],
    startTime: [
        { required: true, message: '请选择开始时间', trigger: 'blur' }
    ],
    endTime: [
        { required: true, message: '请选择到期时间', trigger: 'blur' }
    ],
    pic: [
        { required: true, message: '请选择广告图片', trigger: 'blur' }
    ]
});
// 页面加载时获取数据
onMounted(async () => {
    if (props.isEdit) {
        const res = await getHomeAdvertiseByIdAPI(Number(route.query.id));
        homeAdvertise.value = res.data;
    }
    else {
        homeAdvertise.value = Object.assign({}, defaultHomeAdvertise);
    }
});
// 提交表单
const onSubmit = async () => {
    homeAdvertiseFrom.value.validate(async (valid) => {
        if (valid) {
            await ElMessageBox.confirm('是否提交数据', '提示', {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning'
            });
            if (props.isEdit) {
                await homeAdvertiseUpdateAPI(Number(route.query.id), homeAdvertise.value);
                homeAdvertiseFrom.value.resetFields();
                ElMessage({
                    message: '修改成功',
                    type: 'success',
                    duration: 1000
                });
                router.back();
            }
            else {
                await homeAdvertiseCreateAPI(homeAdvertise.value);
                homeAdvertiseFrom.value.resetFields();
                homeAdvertise.value = Object.assign({}, defaultHomeAdvertise);
                ElMessage({
                    message: '提交成功',
                    type: 'success',
                    duration: 1000
                });
            }
        }
        else {
            ElMessage({
                message: '验证失败',
                type: 'error',
                duration: 1000
            });
        }
    });
};
// 重置表单
const resetForm = () => {
    if (homeAdvertiseFrom.value) {
        homeAdvertiseFrom.value.resetFields();
        homeAdvertise.value = Object.assign({}, defaultHomeAdvertise);
    }
};
const __VLS_ctx = {
    ...{},
    ...{},
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
    ref: "homeAdvertiseFrom",
    model: (__VLS_ctx.homeAdvertise),
    rules: (__VLS_ctx.rules),
    labelWidth: "150px",
}));
const __VLS_9 = __VLS_8({
    ref: "homeAdvertiseFrom",
    model: (__VLS_ctx.homeAdvertise),
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
    label: "广告名称：",
    prop: "name",
}));
const __VLS_17 = __VLS_16({
    label: "广告名称：",
    prop: "name",
}, ...__VLS_functionalComponentArgsRest(__VLS_16));
const { default: __VLS_20 } = __VLS_18.slots;
let __VLS_21;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_22 = __VLS_asFunctionalComponent1(__VLS_21, new __VLS_21({
    modelValue: (__VLS_ctx.homeAdvertise.name),
    ...{ class: "input-width" },
}));
const __VLS_23 = __VLS_22({
    modelValue: (__VLS_ctx.homeAdvertise.name),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_22));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[homeAdvertise, homeAdvertise, rules,];
var __VLS_18;
let __VLS_26;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_27 = __VLS_asFunctionalComponent1(__VLS_26, new __VLS_26({
    label: "广告位置：",
}));
const __VLS_28 = __VLS_27({
    label: "广告位置：",
}, ...__VLS_functionalComponentArgsRest(__VLS_27));
const { default: __VLS_31 } = __VLS_29.slots;
let __VLS_32;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_33 = __VLS_asFunctionalComponent1(__VLS_32, new __VLS_32({
    modelValue: (__VLS_ctx.homeAdvertise.type),
}));
const __VLS_34 = __VLS_33({
    modelValue: (__VLS_ctx.homeAdvertise.type),
}, ...__VLS_functionalComponentArgsRest(__VLS_33));
const { default: __VLS_37 } = __VLS_35.slots;
for (const [type] of __VLS_vFor((__VLS_ctx.typeOptions))) {
    let __VLS_38;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_39 = __VLS_asFunctionalComponent1(__VLS_38, new __VLS_38({
        key: (type.value),
        label: (type.label),
        value: (type.value),
    }));
    const __VLS_40 = __VLS_39({
        key: (type.value),
        label: (type.label),
        value: (type.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_39));
    // @ts-ignore
    [homeAdvertise, typeOptions,];
}
// @ts-ignore
[];
var __VLS_35;
// @ts-ignore
[];
var __VLS_29;
let __VLS_43;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_44 = __VLS_asFunctionalComponent1(__VLS_43, new __VLS_43({
    label: "开始时间：",
    prop: "startTime",
}));
const __VLS_45 = __VLS_44({
    label: "开始时间：",
    prop: "startTime",
}, ...__VLS_functionalComponentArgsRest(__VLS_44));
const { default: __VLS_48 } = __VLS_46.slots;
let __VLS_49;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_50 = __VLS_asFunctionalComponent1(__VLS_49, new __VLS_49({
    type: "datetime",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.homeAdvertise.startTime),
}));
const __VLS_51 = __VLS_50({
    type: "datetime",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.homeAdvertise.startTime),
}, ...__VLS_functionalComponentArgsRest(__VLS_50));
// @ts-ignore
[homeAdvertise,];
var __VLS_46;
let __VLS_54;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    label: "到期时间：",
    prop: "endTime",
}));
const __VLS_56 = __VLS_55({
    label: "到期时间：",
    prop: "endTime",
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
const { default: __VLS_59 } = __VLS_57.slots;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    type: "datetime",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.homeAdvertise.endTime),
}));
const __VLS_62 = __VLS_61({
    type: "datetime",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.homeAdvertise.endTime),
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
// @ts-ignore
[homeAdvertise,];
var __VLS_57;
let __VLS_65;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_66 = __VLS_asFunctionalComponent1(__VLS_65, new __VLS_65({
    label: "上线/下线：",
}));
const __VLS_67 = __VLS_66({
    label: "上线/下线：",
}, ...__VLS_functionalComponentArgsRest(__VLS_66));
const { default: __VLS_70 } = __VLS_68.slots;
let __VLS_71;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_72 = __VLS_asFunctionalComponent1(__VLS_71, new __VLS_71({
    modelValue: (__VLS_ctx.homeAdvertise.status),
}));
const __VLS_73 = __VLS_72({
    modelValue: (__VLS_ctx.homeAdvertise.status),
}, ...__VLS_functionalComponentArgsRest(__VLS_72));
const { default: __VLS_76 } = __VLS_74.slots;
let __VLS_77;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_78 = __VLS_asFunctionalComponent1(__VLS_77, new __VLS_77({
    label: (0),
}));
const __VLS_79 = __VLS_78({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_78));
const { default: __VLS_82 } = __VLS_80.slots;
// @ts-ignore
[homeAdvertise,];
var __VLS_80;
let __VLS_83;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_84 = __VLS_asFunctionalComponent1(__VLS_83, new __VLS_83({
    label: (1),
}));
const __VLS_85 = __VLS_84({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_84));
const { default: __VLS_88 } = __VLS_86.slots;
// @ts-ignore
[];
var __VLS_86;
// @ts-ignore
[];
var __VLS_74;
// @ts-ignore
[];
var __VLS_68;
let __VLS_89;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_90 = __VLS_asFunctionalComponent1(__VLS_89, new __VLS_89({
    label: "广告图片：",
}));
const __VLS_91 = __VLS_90({
    label: "广告图片：",
}, ...__VLS_functionalComponentArgsRest(__VLS_90));
const { default: __VLS_94 } = __VLS_92.slots;
const __VLS_95 = SingleUpload || SingleUpload;
// @ts-ignore
const __VLS_96 = __VLS_asFunctionalComponent1(__VLS_95, new __VLS_95({
    modelValue: (__VLS_ctx.homeAdvertise.pic),
}));
const __VLS_97 = __VLS_96({
    modelValue: (__VLS_ctx.homeAdvertise.pic),
}, ...__VLS_functionalComponentArgsRest(__VLS_96));
// @ts-ignore
[homeAdvertise,];
var __VLS_92;
let __VLS_100;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_101 = __VLS_asFunctionalComponent1(__VLS_100, new __VLS_100({
    label: "排序：",
}));
const __VLS_102 = __VLS_101({
    label: "排序：",
}, ...__VLS_functionalComponentArgsRest(__VLS_101));
const { default: __VLS_105 } = __VLS_103.slots;
let __VLS_106;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_107 = __VLS_asFunctionalComponent1(__VLS_106, new __VLS_106({
    modelValue: (__VLS_ctx.homeAdvertise.sort),
    ...{ class: "input-width" },
}));
const __VLS_108 = __VLS_107({
    modelValue: (__VLS_ctx.homeAdvertise.sort),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_107));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[homeAdvertise,];
var __VLS_103;
let __VLS_111;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_112 = __VLS_asFunctionalComponent1(__VLS_111, new __VLS_111({
    label: "广告链接：",
    prop: "url",
}));
const __VLS_113 = __VLS_112({
    label: "广告链接：",
    prop: "url",
}, ...__VLS_functionalComponentArgsRest(__VLS_112));
const { default: __VLS_116 } = __VLS_114.slots;
let __VLS_117;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_118 = __VLS_asFunctionalComponent1(__VLS_117, new __VLS_117({
    modelValue: (__VLS_ctx.homeAdvertise.url),
    ...{ class: "input-width" },
}));
const __VLS_119 = __VLS_118({
    modelValue: (__VLS_ctx.homeAdvertise.url),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_118));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[homeAdvertise,];
var __VLS_114;
let __VLS_122;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_123 = __VLS_asFunctionalComponent1(__VLS_122, new __VLS_122({
    label: "广告备注：",
}));
const __VLS_124 = __VLS_123({
    label: "广告备注：",
}, ...__VLS_functionalComponentArgsRest(__VLS_123));
const { default: __VLS_127 } = __VLS_125.slots;
let __VLS_128;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_129 = __VLS_asFunctionalComponent1(__VLS_128, new __VLS_128({
    ...{ class: "input-width" },
    type: "textarea",
    rows: (5),
    placeholder: "请输入内容",
    modelValue: (__VLS_ctx.homeAdvertise.note),
}));
const __VLS_130 = __VLS_129({
    ...{ class: "input-width" },
    type: "textarea",
    rows: (5),
    placeholder: "请输入内容",
    modelValue: (__VLS_ctx.homeAdvertise.note),
}, ...__VLS_functionalComponentArgsRest(__VLS_129));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[homeAdvertise,];
var __VLS_125;
let __VLS_133;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_134 = __VLS_asFunctionalComponent1(__VLS_133, new __VLS_133({}));
const __VLS_135 = __VLS_134({}, ...__VLS_functionalComponentArgsRest(__VLS_134));
const { default: __VLS_138 } = __VLS_136.slots;
let __VLS_139;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_140 = __VLS_asFunctionalComponent1(__VLS_139, new __VLS_139({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_141 = __VLS_140({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_140));
let __VLS_144;
const __VLS_145 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.onSubmit();
            // @ts-ignore
            [onSubmit,];
        } });
const { default: __VLS_146 } = __VLS_142.slots;
// @ts-ignore
[];
var __VLS_142;
var __VLS_143;
if (!props.isEdit) {
    let __VLS_147;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_148 = __VLS_asFunctionalComponent1(__VLS_147, new __VLS_147({
        ...{ 'onClick': {} },
    }));
    const __VLS_149 = __VLS_148({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_148));
    let __VLS_152;
    const __VLS_153 = ({ click: {} },
        { onClick: (...[$event]) => {
                if (!(!props.isEdit))
                    return;
                __VLS_ctx.resetForm();
                // @ts-ignore
                [resetForm,];
            } });
    const { default: __VLS_154 } = __VLS_150.slots;
    // @ts-ignore
    [];
    var __VLS_150;
    var __VLS_151;
}
// @ts-ignore
[];
var __VLS_136;
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
const __VLS_export = (await import('vue')).defineComponent({
    props: {
        // 是否为编辑模式
        isEdit: {
            type: Boolean,
            default: false
        }
    },
});
export default {};
