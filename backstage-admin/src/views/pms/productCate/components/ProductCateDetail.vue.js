/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, reactive, onMounted, computed } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getProductCategoryListAPI, productCategoryCreateAPI, productCategoryUpdateByIdAPI, getProductCategoryByIdAPI } from '@/apis/productCate';
import { productAttributeCategoryListWithAttrAPI } from '@/apis/productAttrCate';
import { getProductAttrInfoByCateIdAPI } from '@/apis/productAttr';
import SingleUpload from '@/components/Upload/singleUpload.vue';
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
// 默认商品分类
const defaultProductCate = {
    description: '',
    icon: '',
    keywords: '',
    name: '',
    navStatus: 0,
    parentId: 0,
    productUnit: '',
    showStatus: 0,
    sort: 0
};
// 定义分类id（编辑使用）
const cateId = computed(() => Number(route.query.id));
// 当前操作的商品分类
const productCate = ref(Object.assign({}, defaultProductCate));
// 可选上级分类
const selectProductCateList = ref([]);
// 商品属性ID及属性分类ID集合（筛选属性）
const filterProductAttrList = ref([{ key: 0, value: [] }]);
// 筛选属性级联选择器中的数据
const filterAttrs = ref([]);
// 获取可选上级分类列表
const getSelectProductCateList = async () => {
    const res = await getProductCategoryListAPI(0, { pageSize: 100, pageNum: 1 });
    selectProductCateList.value = res.data.list;
    const noTopProductCate = Object.assign({}, defaultProductCate);
    noTopProductCate.id = 0;
    noTopProductCate.name = '无上级分类';
    selectProductCateList.value.unshift(noTopProductCate);
};
// 获取商品属性分类列表
const getProductAttrCateList = async () => {
    const res = await productAttributeCategoryListWithAttrAPI();
    const list = res.data;
    filterAttrs.value = list.map(item => ({
        label: item.name,
        value: item.id,
        children: item.productAttributeList?.map(it => ({ label: it.name, value: it.id }))
    }));
};
// 组件挂载后执行
onMounted(async () => {
    if (props.isEdit) {
        const res = await getProductCategoryByIdAPI(cateId.value);
        productCate.value = res.data;
        const attrRes = await getProductAttrInfoByCateIdAPI(cateId.value);
        filterProductAttrList.value = attrRes.data.map((item, index) => ({
            key: Date.now() + index,
            value: [item.attributeCategoryId, item.attributeId]
        }));
    }
    else {
        productCate.value = Object.assign({}, defaultProductCate);
    }
    getSelectProductCateList();
    getProductAttrCateList();
});
// 商品分类表单引用
const productCateFrom = ref();
// 商品分类表单校验规则
const rules = reactive({
    name: [
        { required: true, message: '请输入分类名称', trigger: 'blur' },
        { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
    ]
});
// 获取选中的筛选商品属性ID
const getProductAttributeIdList = () => {
    return filterProductAttrList.value.filter(item => item.value && item.value.length === 2)
        .map(item => item.value[1]);
};
// 提交表单
const onSubmit = async () => {
    if (!productCateFrom.value)
        return;
    const valid = await productCateFrom.value.validate().catch(() => false);
    if (valid) {
        await ElMessageBox.confirm('是否提交数据', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        if (props.isEdit) {
            productCate.value.productAttributeIdList = getProductAttributeIdList();
            await productCategoryUpdateByIdAPI(cateId.value, productCate.value);
            ElMessage({
                message: '修改成功',
                type: 'success',
                duration: 1000
            });
            router.back();
        }
        else {
            productCate.value.productAttributeIdList = getProductAttributeIdList();
            await productCategoryCreateAPI(productCate.value);
            productCateFrom.value.resetFields();
            resetForm();
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
};
// 重置表单
const resetForm = () => {
    if (productCateFrom.value) {
        productCateFrom.value.resetFields();
        productCate.value = Object.assign({}, defaultProductCate);
        getSelectProductCateList();
        filterProductAttrList.value = [{ key: 0, value: [] }];
    }
};
// 删除筛选属性
const removeFilterAttr = (filterProductAttr) => {
    if (filterProductAttrList.value.length === 1) {
        ElMessage({
            message: '至少要留一个',
            type: 'warning',
            duration: 1000
        });
        return;
    }
    const index = filterProductAttrList.value.indexOf(filterProductAttr);
    if (index !== -1) {
        filterProductAttrList.value.splice(index, 1);
    }
};
// 添加筛选属性
const handleAddFilterAttr = () => {
    if (filterProductAttrList.value.length === 3) {
        ElMessage({
            message: '最多添加三个',
            type: 'warning',
            duration: 1000
        });
        return;
    }
    filterProductAttrList.value.push({
        value: [],
        key: Date.now()
    });
};
// 过滤器
const filterLabelFilter = (index) => {
    if (index === 0) {
        return '筛选属性：';
    }
    else {
        return '';
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
    ref: "productCateFrom",
    model: (__VLS_ctx.productCate),
    rules: (__VLS_ctx.rules),
    labelWidth: "150px",
}));
const __VLS_9 = __VLS_8({
    ref: "productCateFrom",
    model: (__VLS_ctx.productCate),
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
    label: "分类名称：",
    prop: "name",
}));
const __VLS_17 = __VLS_16({
    label: "分类名称：",
    prop: "name",
}, ...__VLS_functionalComponentArgsRest(__VLS_16));
const { default: __VLS_20 } = __VLS_18.slots;
let __VLS_21;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_22 = __VLS_asFunctionalComponent1(__VLS_21, new __VLS_21({
    modelValue: (__VLS_ctx.productCate.name),
}));
const __VLS_23 = __VLS_22({
    modelValue: (__VLS_ctx.productCate.name),
}, ...__VLS_functionalComponentArgsRest(__VLS_22));
// @ts-ignore
[productCate, productCate, rules,];
var __VLS_18;
let __VLS_26;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_27 = __VLS_asFunctionalComponent1(__VLS_26, new __VLS_26({
    label: "上级分类：",
}));
const __VLS_28 = __VLS_27({
    label: "上级分类：",
}, ...__VLS_functionalComponentArgsRest(__VLS_27));
const { default: __VLS_31 } = __VLS_29.slots;
let __VLS_32;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_33 = __VLS_asFunctionalComponent1(__VLS_32, new __VLS_32({
    modelValue: (__VLS_ctx.productCate.parentId),
    placeholder: "请选择分类",
}));
const __VLS_34 = __VLS_33({
    modelValue: (__VLS_ctx.productCate.parentId),
    placeholder: "请选择分类",
}, ...__VLS_functionalComponentArgsRest(__VLS_33));
const { default: __VLS_37 } = __VLS_35.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.selectProductCateList))) {
    let __VLS_38;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_39 = __VLS_asFunctionalComponent1(__VLS_38, new __VLS_38({
        key: (item.id),
        label: (item.name),
        value: (item.id),
    }));
    const __VLS_40 = __VLS_39({
        key: (item.id),
        label: (item.name),
        value: (item.id),
    }, ...__VLS_functionalComponentArgsRest(__VLS_39));
    // @ts-ignore
    [productCate, selectProductCateList,];
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
    label: "数量单位：",
}));
const __VLS_45 = __VLS_44({
    label: "数量单位：",
}, ...__VLS_functionalComponentArgsRest(__VLS_44));
const { default: __VLS_48 } = __VLS_46.slots;
let __VLS_49;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_50 = __VLS_asFunctionalComponent1(__VLS_49, new __VLS_49({
    modelValue: (__VLS_ctx.productCate.productUnit),
}));
const __VLS_51 = __VLS_50({
    modelValue: (__VLS_ctx.productCate.productUnit),
}, ...__VLS_functionalComponentArgsRest(__VLS_50));
// @ts-ignore
[productCate,];
var __VLS_46;
let __VLS_54;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    label: "排序：",
}));
const __VLS_56 = __VLS_55({
    label: "排序：",
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
const { default: __VLS_59 } = __VLS_57.slots;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    modelValue: (__VLS_ctx.productCate.sort),
}));
const __VLS_62 = __VLS_61({
    modelValue: (__VLS_ctx.productCate.sort),
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
// @ts-ignore
[productCate,];
var __VLS_57;
let __VLS_65;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_66 = __VLS_asFunctionalComponent1(__VLS_65, new __VLS_65({
    label: "是否显示：",
}));
const __VLS_67 = __VLS_66({
    label: "是否显示：",
}, ...__VLS_functionalComponentArgsRest(__VLS_66));
const { default: __VLS_70 } = __VLS_68.slots;
let __VLS_71;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_72 = __VLS_asFunctionalComponent1(__VLS_71, new __VLS_71({
    modelValue: (__VLS_ctx.productCate.showStatus),
}));
const __VLS_73 = __VLS_72({
    modelValue: (__VLS_ctx.productCate.showStatus),
}, ...__VLS_functionalComponentArgsRest(__VLS_72));
const { default: __VLS_76 } = __VLS_74.slots;
let __VLS_77;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_78 = __VLS_asFunctionalComponent1(__VLS_77, new __VLS_77({
    label: (1),
}));
const __VLS_79 = __VLS_78({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_78));
const { default: __VLS_82 } = __VLS_80.slots;
// @ts-ignore
[productCate,];
var __VLS_80;
let __VLS_83;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_84 = __VLS_asFunctionalComponent1(__VLS_83, new __VLS_83({
    label: (0),
}));
const __VLS_85 = __VLS_84({
    label: (0),
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
    label: "是否显示在导航栏：",
}));
const __VLS_91 = __VLS_90({
    label: "是否显示在导航栏：",
}, ...__VLS_functionalComponentArgsRest(__VLS_90));
const { default: __VLS_94 } = __VLS_92.slots;
let __VLS_95;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_96 = __VLS_asFunctionalComponent1(__VLS_95, new __VLS_95({
    modelValue: (__VLS_ctx.productCate.navStatus),
}));
const __VLS_97 = __VLS_96({
    modelValue: (__VLS_ctx.productCate.navStatus),
}, ...__VLS_functionalComponentArgsRest(__VLS_96));
const { default: __VLS_100 } = __VLS_98.slots;
let __VLS_101;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_102 = __VLS_asFunctionalComponent1(__VLS_101, new __VLS_101({
    label: (1),
}));
const __VLS_103 = __VLS_102({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_102));
const { default: __VLS_106 } = __VLS_104.slots;
// @ts-ignore
[productCate,];
var __VLS_104;
let __VLS_107;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_108 = __VLS_asFunctionalComponent1(__VLS_107, new __VLS_107({
    label: (0),
}));
const __VLS_109 = __VLS_108({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_108));
const { default: __VLS_112 } = __VLS_110.slots;
// @ts-ignore
[];
var __VLS_110;
// @ts-ignore
[];
var __VLS_98;
// @ts-ignore
[];
var __VLS_92;
let __VLS_113;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_114 = __VLS_asFunctionalComponent1(__VLS_113, new __VLS_113({
    label: "分类图标：",
}));
const __VLS_115 = __VLS_114({
    label: "分类图标：",
}, ...__VLS_functionalComponentArgsRest(__VLS_114));
const { default: __VLS_118 } = __VLS_116.slots;
const __VLS_119 = SingleUpload || SingleUpload;
// @ts-ignore
const __VLS_120 = __VLS_asFunctionalComponent1(__VLS_119, new __VLS_119({
    modelValue: (__VLS_ctx.productCate.icon),
}));
const __VLS_121 = __VLS_120({
    modelValue: (__VLS_ctx.productCate.icon),
}, ...__VLS_functionalComponentArgsRest(__VLS_120));
// @ts-ignore
[productCate,];
var __VLS_116;
for (const [filterProductAttr, index] of __VLS_vFor((__VLS_ctx.filterProductAttrList))) {
    let __VLS_124;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_125 = __VLS_asFunctionalComponent1(__VLS_124, new __VLS_124({
        label: (__VLS_ctx.filterLabelFilter(index)),
        key: (filterProductAttr.key),
    }));
    const __VLS_126 = __VLS_125({
        label: (__VLS_ctx.filterLabelFilter(index)),
        key: (filterProductAttr.key),
    }, ...__VLS_functionalComponentArgsRest(__VLS_125));
    const { default: __VLS_129 } = __VLS_127.slots;
    let __VLS_130;
    /** @ts-ignore @type { | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader'] | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader']} */
    elCascader;
    // @ts-ignore
    const __VLS_131 = __VLS_asFunctionalComponent1(__VLS_130, new __VLS_130({
        clearable: true,
        modelValue: (filterProductAttr.value),
        options: (__VLS_ctx.filterAttrs),
    }));
    const __VLS_132 = __VLS_131({
        clearable: true,
        modelValue: (filterProductAttr.value),
        options: (__VLS_ctx.filterAttrs),
    }, ...__VLS_functionalComponentArgsRest(__VLS_131));
    let __VLS_135;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_136 = __VLS_asFunctionalComponent1(__VLS_135, new __VLS_135({
        ...{ 'onClick': {} },
        ...{ style: {} },
    }));
    const __VLS_137 = __VLS_136({
        ...{ 'onClick': {} },
        ...{ style: {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_136));
    let __VLS_140;
    const __VLS_141 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.removeFilterAttr(filterProductAttr);
                // @ts-ignore
                [filterProductAttrList, filterLabelFilter, filterAttrs, removeFilterAttr,];
            } });
    const { default: __VLS_142 } = __VLS_138.slots;
    // @ts-ignore
    [];
    var __VLS_138;
    var __VLS_139;
    // @ts-ignore
    [];
    var __VLS_127;
    // @ts-ignore
    [];
}
let __VLS_143;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_144 = __VLS_asFunctionalComponent1(__VLS_143, new __VLS_143({}));
const __VLS_145 = __VLS_144({}, ...__VLS_functionalComponentArgsRest(__VLS_144));
const { default: __VLS_148 } = __VLS_146.slots;
let __VLS_149;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_150 = __VLS_asFunctionalComponent1(__VLS_149, new __VLS_149({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_151 = __VLS_150({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_150));
let __VLS_154;
const __VLS_155 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleAddFilterAttr();
            // @ts-ignore
            [handleAddFilterAttr,];
        } });
const { default: __VLS_156 } = __VLS_152.slots;
// @ts-ignore
[];
var __VLS_152;
var __VLS_153;
// @ts-ignore
[];
var __VLS_146;
let __VLS_157;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_158 = __VLS_asFunctionalComponent1(__VLS_157, new __VLS_157({
    label: "关键词：",
}));
const __VLS_159 = __VLS_158({
    label: "关键词：",
}, ...__VLS_functionalComponentArgsRest(__VLS_158));
const { default: __VLS_162 } = __VLS_160.slots;
let __VLS_163;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_164 = __VLS_asFunctionalComponent1(__VLS_163, new __VLS_163({
    modelValue: (__VLS_ctx.productCate.keywords),
}));
const __VLS_165 = __VLS_164({
    modelValue: (__VLS_ctx.productCate.keywords),
}, ...__VLS_functionalComponentArgsRest(__VLS_164));
// @ts-ignore
[productCate,];
var __VLS_160;
let __VLS_168;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_169 = __VLS_asFunctionalComponent1(__VLS_168, new __VLS_168({
    label: "分类描述：",
}));
const __VLS_170 = __VLS_169({
    label: "分类描述：",
}, ...__VLS_functionalComponentArgsRest(__VLS_169));
const { default: __VLS_173 } = __VLS_171.slots;
let __VLS_174;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_175 = __VLS_asFunctionalComponent1(__VLS_174, new __VLS_174({
    type: "textarea",
    autosize: (true),
    modelValue: (__VLS_ctx.productCate.description),
}));
const __VLS_176 = __VLS_175({
    type: "textarea",
    autosize: (true),
    modelValue: (__VLS_ctx.productCate.description),
}, ...__VLS_functionalComponentArgsRest(__VLS_175));
// @ts-ignore
[productCate,];
var __VLS_171;
let __VLS_179;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_180 = __VLS_asFunctionalComponent1(__VLS_179, new __VLS_179({}));
const __VLS_181 = __VLS_180({}, ...__VLS_functionalComponentArgsRest(__VLS_180));
const { default: __VLS_184 } = __VLS_182.slots;
let __VLS_185;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_186 = __VLS_asFunctionalComponent1(__VLS_185, new __VLS_185({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_187 = __VLS_186({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_186));
let __VLS_190;
const __VLS_191 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.onSubmit();
            // @ts-ignore
            [onSubmit,];
        } });
const { default: __VLS_192 } = __VLS_188.slots;
// @ts-ignore
[];
var __VLS_188;
var __VLS_189;
if (!props.isEdit) {
    let __VLS_193;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_194 = __VLS_asFunctionalComponent1(__VLS_193, new __VLS_193({
        ...{ 'onClick': {} },
    }));
    const __VLS_195 = __VLS_194({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_194));
    let __VLS_198;
    const __VLS_199 = ({ click: {} },
        { onClick: (...[$event]) => {
                if (!(!props.isEdit))
                    return;
                __VLS_ctx.resetForm();
                // @ts-ignore
                [resetForm,];
            } });
    const { default: __VLS_200 } = __VLS_196.slots;
    // @ts-ignore
    [];
    var __VLS_196;
    var __VLS_197;
}
// @ts-ignore
[];
var __VLS_182;
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
