/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted, watch, inject } from 'vue';
import { ElMessage } from 'element-plus';
import { getProductCategoryListWithChildrenAPI } from '@/apis/productCate';
import { getBrandListAPI } from '@/apis/brand';
// 定义属性
const props = defineProps({
    isEdit: {
        type: Boolean,
        default: false
    }
});
// 定义事件
const emit = defineEmits(['next-step']);
// 获取跨层传递的数据
const compProductParam = inject('product-key');
// 级联选择器中当前选中的商品分类，结构为：[父分类ID,分类ID]
const selectProductCateValue = ref([]);
// 级联选择器中的商品分类数据
const productCateOptions = ref([]);
// 获取商品分类列表数据
const getProductCateList = async () => {
    const res = await getProductCategoryListWithChildrenAPI();
    const list = res.data;
    productCateOptions.value = list.map(item => ({
        label: item.name,
        value: item.id,
        children: item.children?.map(it => ({ label: it.name, value: it.id }))
    }));
};
// 选择器中的品牌数据
const brandOptions = ref([]);
// 获取品牌列表数据
const getBrandList = async () => {
    const res = await getBrandListAPI({ pageNum: 1, pageSize: 100 });
    brandOptions.value = res.data.list.map(item => ({ label: item.name, value: item.id }));
};
// 初始化数据
onMounted(async () => {
    await getProductCateList();
    await getBrandList();
    if (props.isEdit) {
        handleEditCreated();
    }
});
// 编辑状态初始化标记位
const hasEditCreated = ref(false);
// 表单引用
const productInfoForm = ref();
// 表单验证规则
const rules = {
    name: [
        { required: true, message: '请输入商品名称', trigger: 'blur' },
        { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
    ],
    subTitle: [{ required: true, message: '请输入商品副标题', trigger: 'blur' }],
    productCategoryId: [{ required: true, message: '请选择商品分类', trigger: 'blur' }],
    brandId: [{ required: true, message: '请选择商品品牌', trigger: 'blur' }],
    description: [{ required: true, message: '请输入商品介绍', trigger: 'blur' }],
    requiredProp: [{ required: true, message: '该项为必填项', trigger: 'blur' }]
};
// 监听选择分类变化
watch(selectProductCateValue, (newValue) => {
    if (newValue && newValue.length === 2) {
        compProductParam.value.productCategoryId = newValue[1];
        compProductParam.value.productCategoryName = getCateNameById(compProductParam.value.productCategoryId);
    }
    else {
        compProductParam.value.productCategoryId = undefined;
        compProductParam.value.productCategoryName = undefined;
    }
});
// 方法定义
const handleEditCreated = () => {
    if (compProductParam.value.productCategoryId) {
        selectProductCateValue.value.push(compProductParam.value.cateParentId);
        selectProductCateValue.value.push(compProductParam.value.productCategoryId);
    }
    hasEditCreated.value = true;
};
// 根据分类ID获取分类名称
const getCateNameById = (id) => {
    for (const item of productCateOptions.value) {
        const child = item.children?.find(child => child.value === id);
        if (child) {
            return child.label;
        }
    }
    return undefined;
};
// 下一步按钮出来事件
const handleNext = async () => {
    if (productInfoForm.value) {
        try {
            const valid = await productInfoForm.value.validate();
            if (valid) {
                emit('next-step');
            }
        }
        catch {
            ElMessage({
                message: '验证失败',
                type: 'error',
                duration: 1000
            });
        }
    }
};
// 处理品牌变化
const handleBrandChange = (val) => {
    const findBrand = brandOptions.value.find(item => item.value === val);
    compProductParam.value.brandName = findBrand?.label;
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
    rules: (__VLS_ctx.rules),
    ref: "productInfoForm",
    labelWidth: "120px",
    ...{ class: "form-inner-container" },
}));
const __VLS_2 = __VLS_1({
    model: (__VLS_ctx.compProductParam),
    rules: (__VLS_ctx.rules),
    ref: "productInfoForm",
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
    label: "商品分类：",
    prop: "productCategoryId",
}));
const __VLS_10 = __VLS_9({
    label: "商品分类：",
    prop: "productCategoryId",
}, ...__VLS_functionalComponentArgsRest(__VLS_9));
const { default: __VLS_13 } = __VLS_11.slots;
let __VLS_14;
/** @ts-ignore @type { | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader'] | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader']} */
elCascader;
// @ts-ignore
const __VLS_15 = __VLS_asFunctionalComponent1(__VLS_14, new __VLS_14({
    modelValue: (__VLS_ctx.selectProductCateValue),
    options: (__VLS_ctx.productCateOptions),
}));
const __VLS_16 = __VLS_15({
    modelValue: (__VLS_ctx.selectProductCateValue),
    options: (__VLS_ctx.productCateOptions),
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
// @ts-ignore
[compProductParam, rules, selectProductCateValue, productCateOptions,];
var __VLS_11;
let __VLS_19;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_20 = __VLS_asFunctionalComponent1(__VLS_19, new __VLS_19({
    label: "商品名称：",
    prop: "name",
}));
const __VLS_21 = __VLS_20({
    label: "商品名称：",
    prop: "name",
}, ...__VLS_functionalComponentArgsRest(__VLS_20));
const { default: __VLS_24 } = __VLS_22.slots;
let __VLS_25;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_26 = __VLS_asFunctionalComponent1(__VLS_25, new __VLS_25({
    modelValue: (__VLS_ctx.compProductParam.name),
}));
const __VLS_27 = __VLS_26({
    modelValue: (__VLS_ctx.compProductParam.name),
}, ...__VLS_functionalComponentArgsRest(__VLS_26));
// @ts-ignore
[compProductParam,];
var __VLS_22;
let __VLS_30;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_31 = __VLS_asFunctionalComponent1(__VLS_30, new __VLS_30({
    label: "副标题：",
    prop: "subTitle",
}));
const __VLS_32 = __VLS_31({
    label: "副标题：",
    prop: "subTitle",
}, ...__VLS_functionalComponentArgsRest(__VLS_31));
const { default: __VLS_35 } = __VLS_33.slots;
let __VLS_36;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_37 = __VLS_asFunctionalComponent1(__VLS_36, new __VLS_36({
    modelValue: (__VLS_ctx.compProductParam.subTitle),
}));
const __VLS_38 = __VLS_37({
    modelValue: (__VLS_ctx.compProductParam.subTitle),
}, ...__VLS_functionalComponentArgsRest(__VLS_37));
// @ts-ignore
[compProductParam,];
var __VLS_33;
let __VLS_41;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_42 = __VLS_asFunctionalComponent1(__VLS_41, new __VLS_41({
    label: "商品品牌：",
    prop: "brandId",
}));
const __VLS_43 = __VLS_42({
    label: "商品品牌：",
    prop: "brandId",
}, ...__VLS_functionalComponentArgsRest(__VLS_42));
const { default: __VLS_46 } = __VLS_44.slots;
let __VLS_47;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({
    ...{ 'onChange': {} },
    modelValue: (__VLS_ctx.compProductParam.brandId),
    placeholder: "请选择品牌",
}));
const __VLS_49 = __VLS_48({
    ...{ 'onChange': {} },
    modelValue: (__VLS_ctx.compProductParam.brandId),
    placeholder: "请选择品牌",
}, ...__VLS_functionalComponentArgsRest(__VLS_48));
let __VLS_52;
const __VLS_53 = ({ change: {} },
    { onChange: (__VLS_ctx.handleBrandChange) });
const { default: __VLS_54 } = __VLS_50.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.brandOptions))) {
    let __VLS_55;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_56 = __VLS_asFunctionalComponent1(__VLS_55, new __VLS_55({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_57 = __VLS_56({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_56));
    // @ts-ignore
    [compProductParam, handleBrandChange, brandOptions,];
}
// @ts-ignore
[];
var __VLS_50;
var __VLS_51;
// @ts-ignore
[];
var __VLS_44;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    label: "商品介绍：",
}));
const __VLS_62 = __VLS_61({
    label: "商品介绍：",
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
const { default: __VLS_65 } = __VLS_63.slots;
let __VLS_66;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_67 = __VLS_asFunctionalComponent1(__VLS_66, new __VLS_66({
    autoSize: (true),
    modelValue: (__VLS_ctx.compProductParam.description),
    type: "textarea",
    placeholder: "请输入内容",
}));
const __VLS_68 = __VLS_67({
    autoSize: (true),
    modelValue: (__VLS_ctx.compProductParam.description),
    type: "textarea",
    placeholder: "请输入内容",
}, ...__VLS_functionalComponentArgsRest(__VLS_67));
// @ts-ignore
[compProductParam,];
var __VLS_63;
let __VLS_71;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_72 = __VLS_asFunctionalComponent1(__VLS_71, new __VLS_71({
    label: "商品货号：",
}));
const __VLS_73 = __VLS_72({
    label: "商品货号：",
}, ...__VLS_functionalComponentArgsRest(__VLS_72));
const { default: __VLS_76 } = __VLS_74.slots;
let __VLS_77;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_78 = __VLS_asFunctionalComponent1(__VLS_77, new __VLS_77({
    modelValue: (__VLS_ctx.compProductParam.productSn),
}));
const __VLS_79 = __VLS_78({
    modelValue: (__VLS_ctx.compProductParam.productSn),
}, ...__VLS_functionalComponentArgsRest(__VLS_78));
// @ts-ignore
[compProductParam,];
var __VLS_74;
let __VLS_82;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_83 = __VLS_asFunctionalComponent1(__VLS_82, new __VLS_82({
    label: "商品售价：",
}));
const __VLS_84 = __VLS_83({
    label: "商品售价：",
}, ...__VLS_functionalComponentArgsRest(__VLS_83));
const { default: __VLS_87 } = __VLS_85.slots;
let __VLS_88;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_89 = __VLS_asFunctionalComponent1(__VLS_88, new __VLS_88({
    modelValue: (__VLS_ctx.compProductParam.price),
}));
const __VLS_90 = __VLS_89({
    modelValue: (__VLS_ctx.compProductParam.price),
}, ...__VLS_functionalComponentArgsRest(__VLS_89));
// @ts-ignore
[compProductParam,];
var __VLS_85;
let __VLS_93;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_94 = __VLS_asFunctionalComponent1(__VLS_93, new __VLS_93({
    label: "市场价：",
}));
const __VLS_95 = __VLS_94({
    label: "市场价：",
}, ...__VLS_functionalComponentArgsRest(__VLS_94));
const { default: __VLS_98 } = __VLS_96.slots;
let __VLS_99;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_100 = __VLS_asFunctionalComponent1(__VLS_99, new __VLS_99({
    modelValue: (__VLS_ctx.compProductParam.originalPrice),
}));
const __VLS_101 = __VLS_100({
    modelValue: (__VLS_ctx.compProductParam.originalPrice),
}, ...__VLS_functionalComponentArgsRest(__VLS_100));
// @ts-ignore
[compProductParam,];
var __VLS_96;
let __VLS_104;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_105 = __VLS_asFunctionalComponent1(__VLS_104, new __VLS_104({
    label: "商品库存：",
}));
const __VLS_106 = __VLS_105({
    label: "商品库存：",
}, ...__VLS_functionalComponentArgsRest(__VLS_105));
const { default: __VLS_109 } = __VLS_107.slots;
let __VLS_110;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_111 = __VLS_asFunctionalComponent1(__VLS_110, new __VLS_110({
    modelValue: (__VLS_ctx.compProductParam.stock),
}));
const __VLS_112 = __VLS_111({
    modelValue: (__VLS_ctx.compProductParam.stock),
}, ...__VLS_functionalComponentArgsRest(__VLS_111));
// @ts-ignore
[compProductParam,];
var __VLS_107;
let __VLS_115;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_116 = __VLS_asFunctionalComponent1(__VLS_115, new __VLS_115({
    label: "计量单位：",
}));
const __VLS_117 = __VLS_116({
    label: "计量单位：",
}, ...__VLS_functionalComponentArgsRest(__VLS_116));
const { default: __VLS_120 } = __VLS_118.slots;
let __VLS_121;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_122 = __VLS_asFunctionalComponent1(__VLS_121, new __VLS_121({
    modelValue: (__VLS_ctx.compProductParam.unit),
}));
const __VLS_123 = __VLS_122({
    modelValue: (__VLS_ctx.compProductParam.unit),
}, ...__VLS_functionalComponentArgsRest(__VLS_122));
// @ts-ignore
[compProductParam,];
var __VLS_118;
let __VLS_126;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_127 = __VLS_asFunctionalComponent1(__VLS_126, new __VLS_126({
    label: "商品重量：",
}));
const __VLS_128 = __VLS_127({
    label: "商品重量：",
}, ...__VLS_functionalComponentArgsRest(__VLS_127));
const { default: __VLS_131 } = __VLS_129.slots;
let __VLS_132;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_133 = __VLS_asFunctionalComponent1(__VLS_132, new __VLS_132({
    modelValue: (__VLS_ctx.compProductParam.weight),
    ...{ style: {} },
}));
const __VLS_134 = __VLS_133({
    modelValue: (__VLS_ctx.compProductParam.weight),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_133));
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
});
// @ts-ignore
[compProductParam,];
var __VLS_129;
let __VLS_137;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_138 = __VLS_asFunctionalComponent1(__VLS_137, new __VLS_137({
    label: "排序",
}));
const __VLS_139 = __VLS_138({
    label: "排序",
}, ...__VLS_functionalComponentArgsRest(__VLS_138));
const { default: __VLS_142 } = __VLS_140.slots;
let __VLS_143;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_144 = __VLS_asFunctionalComponent1(__VLS_143, new __VLS_143({
    modelValue: (__VLS_ctx.compProductParam.sort),
}));
const __VLS_145 = __VLS_144({
    modelValue: (__VLS_ctx.compProductParam.sort),
}, ...__VLS_functionalComponentArgsRest(__VLS_144));
// @ts-ignore
[compProductParam,];
var __VLS_140;
let __VLS_148;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_149 = __VLS_asFunctionalComponent1(__VLS_148, new __VLS_148({}));
const __VLS_150 = __VLS_149({}, ...__VLS_functionalComponentArgsRest(__VLS_149));
const { default: __VLS_153 } = __VLS_151.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_154;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_155 = __VLS_asFunctionalComponent1(__VLS_154, new __VLS_154({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_156 = __VLS_155({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_155));
let __VLS_159;
const __VLS_160 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleNext();
            // @ts-ignore
            [handleNext,];
        } });
const { default: __VLS_161 } = __VLS_157.slots;
// @ts-ignore
[];
var __VLS_157;
var __VLS_158;
// @ts-ignore
[];
var __VLS_151;
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
