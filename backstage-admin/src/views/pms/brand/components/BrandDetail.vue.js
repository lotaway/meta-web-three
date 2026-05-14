/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, reactive, onMounted, computed } from 'vue';
import { createBrandAPI, getBrandAPI, updateBrandAPI } from '@/apis/brand';
import SingleUpload from '@/components/Upload/singleUpload.vue';
import { useRoute, useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
// 获取路由对象
const route = useRoute();
const router = useRouter();
// 定义属性
const props = defineProps({
    // 是否为编辑模式
    isEdit: {
        type: Boolean,
        default: false
    }
});
// 定义品牌id
const brandId = computed(() => Number(route.query.id));
// 默认品牌，添加时使用
const defaultBrand = {
    bigPic: '',
    brandStory: '',
    factoryStatus: 0,
    firstLetter: '',
    logo: '',
    name: '',
    showStatus: 0,
    sort: 0
};
// 定义品牌对象
const brand = ref(Object.assign({}, defaultBrand));
// 品牌表单组件引用
const brandFromRef = ref();
// 品牌表单校验规则
const rules = reactive({
    name: [
        { required: true, message: '请输入品牌名称', trigger: 'blur' },
        { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
    ],
    logo: [
        { required: true, message: '请输入品牌logo', trigger: 'blur' }
    ],
    sort: [
        { type: 'number', message: '排序必须为数字' }
    ],
});
// 组件挂载时加载数据
onMounted(async () => {
    if (props.isEdit) {
        // 编辑模式调用接口获取品牌详情
        const res = await getBrandAPI(brandId.value);
        brand.value = res.data;
    }
    else {
        // 添加模式使用默认品牌
        brand.value = Object.assign({}, defaultBrand);
    }
});
// 处理品牌表单提交
const handleBrandSubmit = () => {
    brandFromRef.value.validate(async (valid) => {
        if (valid) {
            await ElMessageBox.confirm('是否提交数据', '提示', {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning'
            });
            if (props.isEdit) {
                await updateBrandAPI(brandId.value, brand.value);
                brandFromRef.value.resetFields();
                ElMessage({
                    message: '修改成功',
                    type: 'success',
                    duration: 1000
                });
                router.back();
            }
            else {
                await createBrandAPI(brand.value);
                brandFromRef.value.resetFields();
                brand.value = Object.assign({}, defaultBrand);
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
// 处理品牌表单重置
const handleBrandReset = () => {
    brandFromRef.value.resetFields();
    brand.value = Object.assign({}, defaultBrand);
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
    model: (__VLS_ctx.brand),
    rules: (__VLS_ctx.rules),
    ref: "brandFromRef",
    labelWidth: "150px",
}));
const __VLS_9 = __VLS_8({
    model: (__VLS_ctx.brand),
    rules: (__VLS_ctx.rules),
    ref: "brandFromRef",
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
var __VLS_12 = {};
const { default: __VLS_14 } = __VLS_10.slots;
let __VLS_15;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_16 = __VLS_asFunctionalComponent1(__VLS_15, new __VLS_15({
    label: "品牌名称：",
    prop: "name",
}));
const __VLS_17 = __VLS_16({
    label: "品牌名称：",
    prop: "name",
}, ...__VLS_functionalComponentArgsRest(__VLS_16));
const { default: __VLS_20 } = __VLS_18.slots;
let __VLS_21;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_22 = __VLS_asFunctionalComponent1(__VLS_21, new __VLS_21({
    modelValue: (__VLS_ctx.brand.name),
}));
const __VLS_23 = __VLS_22({
    modelValue: (__VLS_ctx.brand.name),
}, ...__VLS_functionalComponentArgsRest(__VLS_22));
// @ts-ignore
[brand, brand, rules,];
var __VLS_18;
let __VLS_26;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_27 = __VLS_asFunctionalComponent1(__VLS_26, new __VLS_26({
    label: "品牌首字母：",
}));
const __VLS_28 = __VLS_27({
    label: "品牌首字母：",
}, ...__VLS_functionalComponentArgsRest(__VLS_27));
const { default: __VLS_31 } = __VLS_29.slots;
let __VLS_32;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_33 = __VLS_asFunctionalComponent1(__VLS_32, new __VLS_32({
    modelValue: (__VLS_ctx.brand.firstLetter),
}));
const __VLS_34 = __VLS_33({
    modelValue: (__VLS_ctx.brand.firstLetter),
}, ...__VLS_functionalComponentArgsRest(__VLS_33));
// @ts-ignore
[brand,];
var __VLS_29;
let __VLS_37;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_38 = __VLS_asFunctionalComponent1(__VLS_37, new __VLS_37({
    label: "品牌LOGO：",
    prop: "logo",
}));
const __VLS_39 = __VLS_38({
    label: "品牌LOGO：",
    prop: "logo",
}, ...__VLS_functionalComponentArgsRest(__VLS_38));
const { default: __VLS_42 } = __VLS_40.slots;
const __VLS_43 = SingleUpload || SingleUpload;
// @ts-ignore
const __VLS_44 = __VLS_asFunctionalComponent1(__VLS_43, new __VLS_43({
    modelValue: (__VLS_ctx.brand.logo),
}));
const __VLS_45 = __VLS_44({
    modelValue: (__VLS_ctx.brand.logo),
}, ...__VLS_functionalComponentArgsRest(__VLS_44));
// @ts-ignore
[brand,];
var __VLS_40;
let __VLS_48;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_49 = __VLS_asFunctionalComponent1(__VLS_48, new __VLS_48({
    label: "品牌专区大图：",
}));
const __VLS_50 = __VLS_49({
    label: "品牌专区大图：",
}, ...__VLS_functionalComponentArgsRest(__VLS_49));
const { default: __VLS_53 } = __VLS_51.slots;
const __VLS_54 = SingleUpload || SingleUpload;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    modelValue: (__VLS_ctx.brand.bigPic),
}));
const __VLS_56 = __VLS_55({
    modelValue: (__VLS_ctx.brand.bigPic),
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
// @ts-ignore
[brand,];
var __VLS_51;
let __VLS_59;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_60 = __VLS_asFunctionalComponent1(__VLS_59, new __VLS_59({
    label: "品牌故事：",
}));
const __VLS_61 = __VLS_60({
    label: "品牌故事：",
}, ...__VLS_functionalComponentArgsRest(__VLS_60));
const { default: __VLS_64 } = __VLS_62.slots;
let __VLS_65;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_66 = __VLS_asFunctionalComponent1(__VLS_65, new __VLS_65({
    placeholder: "请输入内容",
    type: "textarea",
    modelValue: (__VLS_ctx.brand.brandStory),
    autosize: (true),
}));
const __VLS_67 = __VLS_66({
    placeholder: "请输入内容",
    type: "textarea",
    modelValue: (__VLS_ctx.brand.brandStory),
    autosize: (true),
}, ...__VLS_functionalComponentArgsRest(__VLS_66));
// @ts-ignore
[brand,];
var __VLS_62;
let __VLS_70;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_71 = __VLS_asFunctionalComponent1(__VLS_70, new __VLS_70({
    label: "排序：",
    prop: "sort",
}));
const __VLS_72 = __VLS_71({
    label: "排序：",
    prop: "sort",
}, ...__VLS_functionalComponentArgsRest(__VLS_71));
const { default: __VLS_75 } = __VLS_73.slots;
let __VLS_76;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_77 = __VLS_asFunctionalComponent1(__VLS_76, new __VLS_76({
    modelValue: (__VLS_ctx.brand.sort),
    modelModifiers: { number: true, },
}));
const __VLS_78 = __VLS_77({
    modelValue: (__VLS_ctx.brand.sort),
    modelModifiers: { number: true, },
}, ...__VLS_functionalComponentArgsRest(__VLS_77));
// @ts-ignore
[brand,];
var __VLS_73;
let __VLS_81;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_82 = __VLS_asFunctionalComponent1(__VLS_81, new __VLS_81({
    label: "是否显示：",
}));
const __VLS_83 = __VLS_82({
    label: "是否显示：",
}, ...__VLS_functionalComponentArgsRest(__VLS_82));
const { default: __VLS_86 } = __VLS_84.slots;
let __VLS_87;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_88 = __VLS_asFunctionalComponent1(__VLS_87, new __VLS_87({
    modelValue: (__VLS_ctx.brand.showStatus),
}));
const __VLS_89 = __VLS_88({
    modelValue: (__VLS_ctx.brand.showStatus),
}, ...__VLS_functionalComponentArgsRest(__VLS_88));
const { default: __VLS_92 } = __VLS_90.slots;
let __VLS_93;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_94 = __VLS_asFunctionalComponent1(__VLS_93, new __VLS_93({
    label: (1),
}));
const __VLS_95 = __VLS_94({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_94));
const { default: __VLS_98 } = __VLS_96.slots;
// @ts-ignore
[brand,];
var __VLS_96;
let __VLS_99;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_100 = __VLS_asFunctionalComponent1(__VLS_99, new __VLS_99({
    label: (0),
}));
const __VLS_101 = __VLS_100({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_100));
const { default: __VLS_104 } = __VLS_102.slots;
// @ts-ignore
[];
var __VLS_102;
// @ts-ignore
[];
var __VLS_90;
// @ts-ignore
[];
var __VLS_84;
let __VLS_105;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_106 = __VLS_asFunctionalComponent1(__VLS_105, new __VLS_105({
    label: "品牌制造商：",
}));
const __VLS_107 = __VLS_106({
    label: "品牌制造商：",
}, ...__VLS_functionalComponentArgsRest(__VLS_106));
const { default: __VLS_110 } = __VLS_108.slots;
let __VLS_111;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_112 = __VLS_asFunctionalComponent1(__VLS_111, new __VLS_111({
    modelValue: (__VLS_ctx.brand.factoryStatus),
}));
const __VLS_113 = __VLS_112({
    modelValue: (__VLS_ctx.brand.factoryStatus),
}, ...__VLS_functionalComponentArgsRest(__VLS_112));
const { default: __VLS_116 } = __VLS_114.slots;
let __VLS_117;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_118 = __VLS_asFunctionalComponent1(__VLS_117, new __VLS_117({
    label: (1),
}));
const __VLS_119 = __VLS_118({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_118));
const { default: __VLS_122 } = __VLS_120.slots;
// @ts-ignore
[brand,];
var __VLS_120;
let __VLS_123;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_124 = __VLS_asFunctionalComponent1(__VLS_123, new __VLS_123({
    label: (0),
}));
const __VLS_125 = __VLS_124({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_124));
const { default: __VLS_128 } = __VLS_126.slots;
// @ts-ignore
[];
var __VLS_126;
// @ts-ignore
[];
var __VLS_114;
// @ts-ignore
[];
var __VLS_108;
let __VLS_129;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_130 = __VLS_asFunctionalComponent1(__VLS_129, new __VLS_129({}));
const __VLS_131 = __VLS_130({}, ...__VLS_functionalComponentArgsRest(__VLS_130));
const { default: __VLS_134 } = __VLS_132.slots;
let __VLS_135;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_136 = __VLS_asFunctionalComponent1(__VLS_135, new __VLS_135({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_137 = __VLS_136({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_136));
let __VLS_140;
const __VLS_141 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleBrandSubmit();
            // @ts-ignore
            [handleBrandSubmit,];
        } });
const { default: __VLS_142 } = __VLS_138.slots;
// @ts-ignore
[];
var __VLS_138;
var __VLS_139;
if (!props.isEdit) {
    let __VLS_143;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_144 = __VLS_asFunctionalComponent1(__VLS_143, new __VLS_143({
        ...{ 'onClick': {} },
    }));
    const __VLS_145 = __VLS_144({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_144));
    let __VLS_148;
    const __VLS_149 = ({ click: {} },
        { onClick: (...[$event]) => {
                if (!(!props.isEdit))
                    return;
                __VLS_ctx.handleBrandReset();
                // @ts-ignore
                [handleBrandReset,];
            } });
    const { default: __VLS_150 } = __VLS_146.slots;
    // @ts-ignore
    [];
    var __VLS_146;
    var __VLS_147;
}
// @ts-ignore
[];
var __VLS_132;
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
