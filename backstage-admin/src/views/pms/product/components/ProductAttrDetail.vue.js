/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, reactive, computed, onMounted, inject, watch } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getProductAttributeCategoryListAPI as fetchProductAttrCateList } from '@/apis/productAttrCate';
import { getProductAttributeListAPI } from '@/apis/productAttr';
import SingleUpload from '@/components/Upload/singleUpload.vue';
import MultiUpload from '@/components/Upload/multiUpload.vue';
import Tinymce from '@/components/Tinymce/index.vue';
// 定义属性
const props = defineProps({
    isEdit: {
        type: Boolean,
        default: false
    }
});
// 定义事件
const emit = defineEmits(['prev-step', 'next-step']);
// 获取跨层传递的数据
const compProductParam = inject('product-key');
// 模板引用
const productAttrForm = ref();
// 响应式数据
const state = reactive({
    // 编辑模式时是否初始化成功
    hasEditCreated: false,
    // 商品属性分类下拉选项
    productAttributeCategoryOptions: [],
    // 选中的商品规格
    selectProductAttr: [],
    // 选中的商品参数
    selectProductParam: [],
    // 选中的商品属性图片
    selectProductAttrPics: [],
    // 可手动添加的商品属性
    addProductAttrValue: '',
    // 商品富文本详情激活类型
    activeHtmlName: 'pc'
});
// 商品的ID
const productId = computed(() => {
    return compProductParam.value.id;
});
// 由于compProductParam数据为异步加载，需要监听其变化来初始化数据
watch(productId, (newVal, oldVal) => {
    if (props.isEdit) {
        handleEditCreated();
    }
    console.log("attr", newVal, oldVal);
});
// 获取商品属性分类
const getProductAttrCateList = async () => {
    const param = { pageNum: 1, pageSize: 100 };
    const res = await fetchProductAttrCateList(param);
    state.productAttributeCategoryOptions = res.data.list.map(item => ({
        label: item.name,
        value: item.id
    }));
};
onMounted(async () => {
    await getProductAttrCateList();
});
// 是否有商品属性图片
const hasAttrPic = computed(() => {
    if (state.selectProductAttrPics.length < 1) {
        return false;
    }
    return true;
});
// 商品的主图和画册图片
const selectProductPics = computed({
    get: function () {
        const pics = [];
        if (!compProductParam.value.pic || compProductParam.value.pic === '') {
            return pics;
        }
        pics.push(compProductParam.value.pic);
        if (!compProductParam.value.albumPics || compProductParam.value.albumPics === '') {
            return pics;
        }
        const albumPics = compProductParam.value.albumPics.split(',');
        pics.push(...albumPics);
        return pics;
    },
    set: function (newValue) {
        if (!newValue || newValue.length === 0) {
            compProductParam.value.pic = undefined;
            compProductParam.value.albumPics = undefined;
        }
        else {
            compProductParam.value.pic = newValue[0];
            compProductParam.value.albumPics = '';
            if (newValue.length > 1) {
                for (let i = 1; i < newValue.length; i++) {
                    compProductParam.value.albumPics += newValue[i];
                    if (i !== newValue.length - 1) {
                        compProductParam.value.albumPics += ',';
                    }
                }
            }
        }
    }
});
// 处理编辑模式初始化的数据
const handleEditCreated = () => {
    if (compProductParam.value.productAttributeCategoryId) {
        handleProductAttrChange(compProductParam.value.productAttributeCategoryId);
    }
    state.hasEditCreated = true;
};
/**
 * 根据商品属性分类ID获取规格和参数
 * @param type 0:规格;1:参数
 * @param cid 商品属性分类ID
 */
const getProductAttrList = async (type, cid) => {
    const param = { pageNum: 1, pageSize: 100, type: type };
    const res = await getProductAttributeListAPI(cid, param);
    const list = res.data.list;
    if (type === 0) {
        state.selectProductAttr = [];
        for (let i = 0; i < list.length; i++) {
            const item = list[i];
            let options = [];
            let values = [];
            if (props.isEdit) {
                if (item.handAddStatus === 1) {
                    // 编辑状态下获取手动添加编辑属性
                    options = getEditAttrOptions(item.id);
                }
                // 编辑状态下获取选中属性
                values = getEditAttrValues(i);
            }
            state.selectProductAttr.push({
                id: item.id,
                name: item.name,
                handAddStatus: item.handAddStatus,
                inputList: item.inputList,
                values: values,
                options: options
            });
        }
        if (props.isEdit) {
            // 编辑模式下刷新商品属性图片
            refreshProductAttrPics();
        }
    }
    else {
        state.selectProductParam = [];
        for (let i = 0; i < list.length; i++) {
            const item = list[i];
            let value = undefined;
            if (props.isEdit) {
                // 编辑模式下获取参数属性
                value = getEditParamValue(item.id);
            }
            state.selectProductParam.push({
                id: item.id,
                name: item.name,
                value: value,
                inputType: item.inputType,
                inputList: item.inputList
            });
        }
    }
};
// 获取设置的可手动添加属性值
const getEditAttrOptions = (id) => {
    const attrValueList = compProductParam.value.productAttributeValueList;
    return attrValueList?.find(item => item.productAttributeId === id)?.value?.split(',');
};
// 获取选中的规格值
const getEditAttrValues = (index) => {
    const skuList = compProductParam.value.skuStockList;
    const values = new Set();
    for (let i = 0; i < skuList.length; i++) {
        const sku = skuList[i];
        const spData = JSON.parse(sku.spData);
        if (spData && spData.length > index) {
            values.add(spData[index].value);
        }
    }
    return Array.from(values);
};
// 获取属性的值
const getEditParamValue = (id) => {
    return compProductParam.value.productAttributeValueList?.find(item => item.productAttributeId === id)?.value;
};
const handleProductAttrChange = (value) => {
    getProductAttrList(0, value);
    getProductAttrList(1, value);
};
const getInputListArr = (inputList) => {
    return inputList.split(',');
};
const handleAddProductAttrValue = (idx) => {
    const options = state.selectProductAttr[idx].options;
    if (state.addProductAttrValue == null || state.addProductAttrValue == '') {
        ElMessage({
            message: '属性值不能为空',
            type: 'warning',
            duration: 1000
        });
        return;
    }
    if (options.indexOf(state.addProductAttrValue) !== -1) {
        ElMessage({
            message: '属性值不能重复',
            type: 'warning',
            duration: 1000
        });
        return;
    }
    options?.push(state.addProductAttrValue);
    state.addProductAttrValue = '';
};
const handleRemoveProductAttrValue = (idx, index) => {
    state.selectProductAttr[idx]?.options?.splice(index, 1);
};
const getProductSkuSp = (row, index) => {
    const spData = JSON.parse(row.spData);
    if (spData != null && index < spData.length) {
        return spData[index].value;
    }
    else {
        return null;
    }
};
const handleRefreshProductSkuList = async () => {
    try {
        await ElMessageBox.confirm('刷新列表将导致sku信息重新生成，是否要刷新', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        refreshProductAttrPics();
        refreshProductSkuList();
    }
    catch {
        // 用户取消操作
    }
};
const handleSyncProductSkuPrice = async () => {
    try {
        await ElMessageBox.confirm('将同步第一个sku的价格到所有sku,是否继续', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        if (compProductParam.value.skuStockList && compProductParam.value.skuStockList.length > 0) {
            let tempSkuList = [];
            tempSkuList = tempSkuList.concat(tempSkuList, compProductParam.value.skuStockList);
            const price = compProductParam.value.skuStockList[0].price;
            for (let i = 0; i < tempSkuList.length; i++) {
                tempSkuList[i].price = price;
            }
            compProductParam.value.skuStockList = [];
            compProductParam.value.skuStockList = compProductParam.value.skuStockList.concat(compProductParam.value.skuStockList, tempSkuList);
        }
    }
    catch {
        // 用户取消操作
    }
};
const handleSyncProductSkuStock = async () => {
    try {
        await ElMessageBox.confirm('将同步第一个sku的库存到所有sku,是否继续', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        if (compProductParam.value.skuStockList && compProductParam.value.skuStockList.length > 0) {
            let tempSkuList = [];
            tempSkuList = tempSkuList.concat(tempSkuList, compProductParam.value.skuStockList);
            const stock = compProductParam.value.skuStockList[0].stock;
            const lowStock = compProductParam.value.skuStockList[0].lowStock;
            for (let i = 0; i < tempSkuList.length; i++) {
                tempSkuList[i].stock = stock;
                tempSkuList[i].lowStock = lowStock;
            }
            compProductParam.value.skuStockList = [];
            compProductParam.value.skuStockList = compProductParam.value.skuStockList.concat(compProductParam.value.skuStockList, tempSkuList);
        }
    }
    catch {
        // 用户取消操作
    }
};
const refreshProductSkuList = () => {
    compProductParam.value.skuStockList = [];
    const skuList = compProductParam.value.skuStockList;
    // 只有一个属性时
    if (state.selectProductAttr.length === 1) {
        const attr = state.selectProductAttr[0];
        for (let i = 0; i < attr.values.length; i++) {
            skuList.push({
                spData: JSON.stringify([{ key: attr.name, value: attr.values[i] }])
            });
        }
    }
    else if (state.selectProductAttr.length === 2) {
        const attr0 = state.selectProductAttr[0];
        const attr1 = state.selectProductAttr[1];
        for (let i = 0; i < attr0.values.length; i++) {
            if (attr1.values.length === 0) {
                skuList.push({
                    spData: JSON.stringify([{ key: attr0.name, value: attr0.values[i] }])
                });
                continue;
            }
            for (let j = 0; j < attr1.values.length; j++) {
                const spData = [];
                spData.push({ key: attr0.name, value: attr0.values[i] });
                spData.push({ key: attr1.name, value: attr1.values[j] });
                skuList.push({
                    spData: JSON.stringify(spData)
                });
            }
        }
    }
    else {
        const attr0 = state.selectProductAttr[0];
        const attr1 = state.selectProductAttr[1];
        const attr2 = state.selectProductAttr[2];
        for (let i = 0; i < attr0.values.length; i++) {
            if (attr1.values.length === 0) {
                skuList.push({
                    spData: JSON.stringify([{ key: attr0.name, value: attr0.values[i] }])
                });
                continue;
            }
            for (let j = 0; j < attr1.values.length; j++) {
                if (attr2.values.length === 0) {
                    const spData = [];
                    spData.push({ key: attr0.name, value: attr0.values[i] });
                    spData.push({ key: attr1.name, value: attr1.values[j] });
                    skuList.push({
                        spData: JSON.stringify(spData)
                    });
                    continue;
                }
                for (let k = 0; k < attr2.values.length; k++) {
                    const spData = [];
                    spData.push({ key: attr0.name, value: attr0.values[i] });
                    spData.push({ key: attr1.name, value: attr1.values[j] });
                    spData.push({ key: attr2.name, value: attr2.values[k] });
                    skuList.push({
                        spData: JSON.stringify(spData)
                    });
                }
            }
        }
    }
};
const refreshProductAttrPics = () => {
    state.selectProductAttrPics = [];
    if (state.selectProductAttr && state.selectProductAttr.length >= 1) {
        const values = state.selectProductAttr[0].values;
        for (let i = 0; i < values.length; i++) {
            let pic = undefined;
            if (props.isEdit) {
                // 编辑状态下获取图片
                pic = getProductSkuPic(values[i]);
            }
            state.selectProductAttrPics.push({ name: values[i], pic: pic });
        }
    }
};
// 获取商品相关属性的图片
const getProductSkuPic = (name) => {
    return compProductParam.value.skuStockList?.find(item => {
        const spData = JSON.parse(item.spData);
        return name === spData[0].value;
    })?.pic;
};
// 合并商品属性
const mergeProductAttrValue = () => {
    compProductParam.value.productAttributeValueList = [];
    for (let i = 0; i < state.selectProductAttr.length; i++) {
        const attr = state.selectProductAttr[i];
        if (attr.handAddStatus === 1 && attr.options != null && attr.options.length > 0) {
            compProductParam.value.productAttributeValueList.push({
                productAttributeId: attr.id,
                value: getOptionStr(attr.options)
            });
        }
    }
    for (let i = 0; i < state.selectProductParam.length; i++) {
        const param = state.selectProductParam[i];
        compProductParam.value.productAttributeValueList.push({
            productAttributeId: param.id,
            value: param.value
        });
    }
};
// 合并商品属性图片
const mergeProductAttrPics = () => {
    for (let i = 0; i < state.selectProductAttrPics.length; i++) {
        const skuStockList = compProductParam.value.skuStockList;
        for (let j = 0; j < skuStockList.length; j++) {
            const spData = JSON.parse(skuStockList[j].spData);
            if (spData[0].value === state.selectProductAttrPics[i].name) {
                skuStockList[j].pic = state.selectProductAttrPics[i].pic;
            }
        }
    }
};
const getOptionStr = (arr) => {
    let str = '';
    for (let i = 0; i < arr.length; i++) {
        str += arr[i];
        if (i != arr.length - 1) {
            str += ',';
        }
    }
    return str;
};
const handleRemoveProductSku = (index) => {
    const list = compProductParam.value.skuStockList;
    if (list.length === 1) {
        list.pop();
    }
    else {
        list.splice(index, 1);
    }
};
const getParamInputList = (inputList) => {
    return inputList.split(',');
};
const handlePrev = () => {
    emit('prev-step');
};
const handleNext = () => {
    mergeProductAttrValue();
    mergeProductAttrPics();
    emit('next-step');
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
    ref: "productAttrForm",
    labelWidth: "120px",
    ...{ class: "form-inner-container" },
}));
const __VLS_2 = __VLS_1({
    model: (__VLS_ctx.compProductParam),
    ref: "productAttrForm",
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
    label: "属性类型：",
}));
const __VLS_10 = __VLS_9({
    label: "属性类型：",
}, ...__VLS_functionalComponentArgsRest(__VLS_9));
const { default: __VLS_13 } = __VLS_11.slots;
let __VLS_14;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_15 = __VLS_asFunctionalComponent1(__VLS_14, new __VLS_14({
    ...{ 'onChange': {} },
    modelValue: (__VLS_ctx.compProductParam.productAttributeCategoryId),
    placeholder: "请选择属性类型",
}));
const __VLS_16 = __VLS_15({
    ...{ 'onChange': {} },
    modelValue: (__VLS_ctx.compProductParam.productAttributeCategoryId),
    placeholder: "请选择属性类型",
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
let __VLS_19;
const __VLS_20 = ({ change: {} },
    { onChange: (__VLS_ctx.handleProductAttrChange) });
const { default: __VLS_21 } = __VLS_17.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.state.productAttributeCategoryOptions))) {
    let __VLS_22;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_23 = __VLS_asFunctionalComponent1(__VLS_22, new __VLS_22({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_24 = __VLS_23({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_23));
    // @ts-ignore
    [compProductParam, compProductParam, handleProductAttrChange, state,];
}
// @ts-ignore
[];
var __VLS_17;
var __VLS_18;
// @ts-ignore
[];
var __VLS_11;
let __VLS_27;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_28 = __VLS_asFunctionalComponent1(__VLS_27, new __VLS_27({
    label: "商品规格：",
}));
const __VLS_29 = __VLS_28({
    label: "商品规格：",
}, ...__VLS_functionalComponentArgsRest(__VLS_28));
const { default: __VLS_32 } = __VLS_30.slots;
let __VLS_33;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_34 = __VLS_asFunctionalComponent1(__VLS_33, new __VLS_33({
    shadow: "never",
    ...{ class: "cardBg" },
}));
const __VLS_35 = __VLS_34({
    shadow: "never",
    ...{ class: "cardBg" },
}, ...__VLS_functionalComponentArgsRest(__VLS_34));
/** @type {__VLS_StyleScopedClasses['cardBg']} */ ;
const { default: __VLS_38 } = __VLS_36.slots;
for (const [productAttr, idx] of __VLS_vFor((__VLS_ctx.state.selectProductAttr))) {
    __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
        key: ('productAttr' + idx),
    });
    (productAttr.name);
    if (productAttr.handAddStatus === 0) {
        let __VLS_39;
        /** @ts-ignore @type { | typeof __VLS_components.elCheckboxGroup | typeof __VLS_components.ElCheckboxGroup | typeof __VLS_components['el-checkbox-group'] | typeof __VLS_components.elCheckboxGroup | typeof __VLS_components.ElCheckboxGroup | typeof __VLS_components['el-checkbox-group']} */
        elCheckboxGroup;
        // @ts-ignore
        const __VLS_40 = __VLS_asFunctionalComponent1(__VLS_39, new __VLS_39({
            modelValue: (__VLS_ctx.state.selectProductAttr[idx].values),
        }));
        const __VLS_41 = __VLS_40({
            modelValue: (__VLS_ctx.state.selectProductAttr[idx].values),
        }, ...__VLS_functionalComponentArgsRest(__VLS_40));
        const { default: __VLS_44 } = __VLS_42.slots;
        for (const [item] of __VLS_vFor((__VLS_ctx.getInputListArr(productAttr.inputList)))) {
            let __VLS_45;
            /** @ts-ignore @type { | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox'] | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox']} */
            elCheckbox;
            // @ts-ignore
            const __VLS_46 = __VLS_asFunctionalComponent1(__VLS_45, new __VLS_45({
                label: (item),
                key: (item),
                ...{ class: "littleMarginLeft" },
            }));
            const __VLS_47 = __VLS_46({
                label: (item),
                key: (item),
                ...{ class: "littleMarginLeft" },
            }, ...__VLS_functionalComponentArgsRest(__VLS_46));
            /** @type {__VLS_StyleScopedClasses['littleMarginLeft']} */ ;
            // @ts-ignore
            [state, state, getInputListArr,];
        }
        // @ts-ignore
        [];
        var __VLS_42;
    }
    else {
        __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
        let __VLS_50;
        /** @ts-ignore @type { | typeof __VLS_components.elCheckboxGroup | typeof __VLS_components.ElCheckboxGroup | typeof __VLS_components['el-checkbox-group'] | typeof __VLS_components.elCheckboxGroup | typeof __VLS_components.ElCheckboxGroup | typeof __VLS_components['el-checkbox-group']} */
        elCheckboxGroup;
        // @ts-ignore
        const __VLS_51 = __VLS_asFunctionalComponent1(__VLS_50, new __VLS_50({
            modelValue: (__VLS_ctx.state.selectProductAttr[idx].values),
        }));
        const __VLS_52 = __VLS_51({
            modelValue: (__VLS_ctx.state.selectProductAttr[idx].values),
        }, ...__VLS_functionalComponentArgsRest(__VLS_51));
        const { default: __VLS_55 } = __VLS_53.slots;
        for (const [item, index] of __VLS_vFor((__VLS_ctx.state.selectProductAttr[idx].options))) {
            __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
                ...{ style: {} },
                ...{ class: "littleMarginLeft" },
                key: ('optons' + index),
            });
            /** @type {__VLS_StyleScopedClasses['littleMarginLeft']} */ ;
            let __VLS_56;
            /** @ts-ignore @type { | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox'] | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox']} */
            elCheckbox;
            // @ts-ignore
            const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
                label: (item),
                key: (item),
            }));
            const __VLS_58 = __VLS_57({
                label: (item),
                key: (item),
            }, ...__VLS_functionalComponentArgsRest(__VLS_57));
            let __VLS_61;
            /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
            elButton;
            // @ts-ignore
            const __VLS_62 = __VLS_asFunctionalComponent1(__VLS_61, new __VLS_61({
                ...{ 'onClick': {} },
                type: "primary",
                link: true,
                ...{ class: "littleMarginLeft" },
            }));
            const __VLS_63 = __VLS_62({
                ...{ 'onClick': {} },
                type: "primary",
                link: true,
                ...{ class: "littleMarginLeft" },
            }, ...__VLS_functionalComponentArgsRest(__VLS_62));
            let __VLS_66;
            const __VLS_67 = ({ click: {} },
                { onClick: (...[$event]) => {
                        if (!!(productAttr.handAddStatus === 0))
                            return;
                        __VLS_ctx.handleRemoveProductAttrValue(idx, index);
                        // @ts-ignore
                        [state, state, handleRemoveProductAttrValue,];
                    } });
            /** @type {__VLS_StyleScopedClasses['littleMarginLeft']} */ ;
            const { default: __VLS_68 } = __VLS_64.slots;
            // @ts-ignore
            [];
            var __VLS_64;
            var __VLS_65;
            // @ts-ignore
            [];
        }
        // @ts-ignore
        [];
        var __VLS_53;
        let __VLS_69;
        /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
        elInput;
        // @ts-ignore
        const __VLS_70 = __VLS_asFunctionalComponent1(__VLS_69, new __VLS_69({
            modelValue: (__VLS_ctx.state.addProductAttrValue),
            ...{ style: {} },
            clearable: true,
        }));
        const __VLS_71 = __VLS_70({
            modelValue: (__VLS_ctx.state.addProductAttrValue),
            ...{ style: {} },
            clearable: true,
        }, ...__VLS_functionalComponentArgsRest(__VLS_70));
        let __VLS_74;
        /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
        elButton;
        // @ts-ignore
        const __VLS_75 = __VLS_asFunctionalComponent1(__VLS_74, new __VLS_74({
            ...{ 'onClick': {} },
            ...{ class: "littleMarginLeft" },
        }));
        const __VLS_76 = __VLS_75({
            ...{ 'onClick': {} },
            ...{ class: "littleMarginLeft" },
        }, ...__VLS_functionalComponentArgsRest(__VLS_75));
        let __VLS_79;
        const __VLS_80 = ({ click: {} },
            { onClick: (...[$event]) => {
                    if (!!(productAttr.handAddStatus === 0))
                        return;
                    __VLS_ctx.handleAddProductAttrValue(idx);
                    // @ts-ignore
                    [state, handleAddProductAttrValue,];
                } });
        /** @type {__VLS_StyleScopedClasses['littleMarginLeft']} */ ;
        const { default: __VLS_81 } = __VLS_77.slots;
        // @ts-ignore
        [];
        var __VLS_77;
        var __VLS_78;
    }
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_36;
let __VLS_82;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_83 = __VLS_asFunctionalComponent1(__VLS_82, new __VLS_82({
    ...{ style: {} },
    data: (__VLS_ctx.compProductParam.skuStockList),
    border: true,
}));
const __VLS_84 = __VLS_83({
    ...{ style: {} },
    data: (__VLS_ctx.compProductParam.skuStockList),
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_83));
const { default: __VLS_87 } = __VLS_85.slots;
for (const [item, index] of __VLS_vFor((__VLS_ctx.state.selectProductAttr))) {
    let __VLS_88;
    /** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
    elTableColumn;
    // @ts-ignore
    const __VLS_89 = __VLS_asFunctionalComponent1(__VLS_88, new __VLS_88({
        label: (item.name),
        key: (item.id),
        align: "center",
    }));
    const __VLS_90 = __VLS_89({
        label: (item.name),
        key: (item.id),
        align: "center",
    }, ...__VLS_functionalComponentArgsRest(__VLS_89));
    const { default: __VLS_93 } = __VLS_91.slots;
    {
        const { default: __VLS_94 } = __VLS_91.slots;
        const [scope] = __VLS_vSlot(__VLS_94);
        (__VLS_ctx.getProductSkuSp(scope.row, index));
        // @ts-ignore
        [compProductParam, state, getProductSkuSp,];
    }
    // @ts-ignore
    [];
    var __VLS_91;
    // @ts-ignore
    [];
}
let __VLS_95;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_96 = __VLS_asFunctionalComponent1(__VLS_95, new __VLS_95({
    label: "销售价格",
    width: "100",
    align: "center",
}));
const __VLS_97 = __VLS_96({
    label: "销售价格",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_96));
const { default: __VLS_100 } = __VLS_98.slots;
{
    const { default: __VLS_101 } = __VLS_98.slots;
    const [scope] = __VLS_vSlot(__VLS_101);
    let __VLS_102;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_103 = __VLS_asFunctionalComponent1(__VLS_102, new __VLS_102({
        modelValue: (scope.row.price),
    }));
    const __VLS_104 = __VLS_103({
        modelValue: (scope.row.price),
    }, ...__VLS_functionalComponentArgsRest(__VLS_103));
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_98;
let __VLS_107;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_108 = __VLS_asFunctionalComponent1(__VLS_107, new __VLS_107({
    label: "促销价格",
    width: "100",
    align: "center",
}));
const __VLS_109 = __VLS_108({
    label: "促销价格",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_108));
const { default: __VLS_112 } = __VLS_110.slots;
{
    const { default: __VLS_113 } = __VLS_110.slots;
    const [scope] = __VLS_vSlot(__VLS_113);
    let __VLS_114;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_115 = __VLS_asFunctionalComponent1(__VLS_114, new __VLS_114({
        modelValue: (scope.row.promotionPrice),
    }));
    const __VLS_116 = __VLS_115({
        modelValue: (scope.row.promotionPrice),
    }, ...__VLS_functionalComponentArgsRest(__VLS_115));
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_110;
let __VLS_119;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_120 = __VLS_asFunctionalComponent1(__VLS_119, new __VLS_119({
    label: "商品库存",
    width: "80",
    align: "center",
}));
const __VLS_121 = __VLS_120({
    label: "商品库存",
    width: "80",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_120));
const { default: __VLS_124 } = __VLS_122.slots;
{
    const { default: __VLS_125 } = __VLS_122.slots;
    const [scope] = __VLS_vSlot(__VLS_125);
    let __VLS_126;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_127 = __VLS_asFunctionalComponent1(__VLS_126, new __VLS_126({
        modelValue: (scope.row.stock),
    }));
    const __VLS_128 = __VLS_127({
        modelValue: (scope.row.stock),
    }, ...__VLS_functionalComponentArgsRest(__VLS_127));
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_122;
let __VLS_131;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_132 = __VLS_asFunctionalComponent1(__VLS_131, new __VLS_131({
    label: "库存预警值",
    width: "80",
    align: "center",
}));
const __VLS_133 = __VLS_132({
    label: "库存预警值",
    width: "80",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_132));
const { default: __VLS_136 } = __VLS_134.slots;
{
    const { default: __VLS_137 } = __VLS_134.slots;
    const [scope] = __VLS_vSlot(__VLS_137);
    let __VLS_138;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_139 = __VLS_asFunctionalComponent1(__VLS_138, new __VLS_138({
        modelValue: (scope.row.lowStock),
    }));
    const __VLS_140 = __VLS_139({
        modelValue: (scope.row.lowStock),
    }, ...__VLS_functionalComponentArgsRest(__VLS_139));
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_134;
let __VLS_143;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_144 = __VLS_asFunctionalComponent1(__VLS_143, new __VLS_143({
    label: "SKU编号",
    width: "160",
    align: "center",
}));
const __VLS_145 = __VLS_144({
    label: "SKU编号",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_144));
const { default: __VLS_148 } = __VLS_146.slots;
{
    const { default: __VLS_149 } = __VLS_146.slots;
    const [scope] = __VLS_vSlot(__VLS_149);
    let __VLS_150;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_151 = __VLS_asFunctionalComponent1(__VLS_150, new __VLS_150({
        modelValue: (scope.row.skuCode),
    }));
    const __VLS_152 = __VLS_151({
        modelValue: (scope.row.skuCode),
    }, ...__VLS_functionalComponentArgsRest(__VLS_151));
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_146;
let __VLS_155;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_156 = __VLS_asFunctionalComponent1(__VLS_155, new __VLS_155({
    label: "操作",
    width: "80",
    align: "center",
}));
const __VLS_157 = __VLS_156({
    label: "操作",
    width: "80",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_156));
const { default: __VLS_160 } = __VLS_158.slots;
{
    const { default: __VLS_161 } = __VLS_158.slots;
    const [scope] = __VLS_vSlot(__VLS_161);
    let __VLS_162;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_163 = __VLS_asFunctionalComponent1(__VLS_162, new __VLS_162({
        ...{ 'onClick': {} },
        type: "primary",
        link: true,
    }));
    const __VLS_164 = __VLS_163({
        ...{ 'onClick': {} },
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_163));
    let __VLS_167;
    const __VLS_168 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleRemoveProductSku(scope.$index);
                // @ts-ignore
                [handleRemoveProductSku,];
            } });
    const { default: __VLS_169 } = __VLS_165.slots;
    // @ts-ignore
    [];
    var __VLS_165;
    var __VLS_166;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_158;
// @ts-ignore
[];
var __VLS_85;
let __VLS_170;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_171 = __VLS_asFunctionalComponent1(__VLS_170, new __VLS_170({
    ...{ 'onClick': {} },
    type: "primary",
    ...{ style: {} },
}));
const __VLS_172 = __VLS_171({
    ...{ 'onClick': {} },
    type: "primary",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_171));
let __VLS_175;
const __VLS_176 = ({ click: {} },
    { onClick: (__VLS_ctx.handleRefreshProductSkuList) });
const { default: __VLS_177 } = __VLS_173.slots;
// @ts-ignore
[handleRefreshProductSkuList,];
var __VLS_173;
var __VLS_174;
let __VLS_178;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_179 = __VLS_asFunctionalComponent1(__VLS_178, new __VLS_178({
    ...{ 'onClick': {} },
    type: "primary",
    ...{ style: {} },
}));
const __VLS_180 = __VLS_179({
    ...{ 'onClick': {} },
    type: "primary",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_179));
let __VLS_183;
const __VLS_184 = ({ click: {} },
    { onClick: (__VLS_ctx.handleSyncProductSkuPrice) });
const { default: __VLS_185 } = __VLS_181.slots;
// @ts-ignore
[handleSyncProductSkuPrice,];
var __VLS_181;
var __VLS_182;
let __VLS_186;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_187 = __VLS_asFunctionalComponent1(__VLS_186, new __VLS_186({
    ...{ 'onClick': {} },
    type: "primary",
    ...{ style: {} },
}));
const __VLS_188 = __VLS_187({
    ...{ 'onClick': {} },
    type: "primary",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_187));
let __VLS_191;
const __VLS_192 = ({ click: {} },
    { onClick: (__VLS_ctx.handleSyncProductSkuStock) });
const { default: __VLS_193 } = __VLS_189.slots;
// @ts-ignore
[handleSyncProductSkuStock,];
var __VLS_189;
var __VLS_190;
// @ts-ignore
[];
var __VLS_30;
if (__VLS_ctx.hasAttrPic) {
    let __VLS_194;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_195 = __VLS_asFunctionalComponent1(__VLS_194, new __VLS_194({
        label: "属性图片：",
    }));
    const __VLS_196 = __VLS_195({
        label: "属性图片：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_195));
    const { default: __VLS_199 } = __VLS_197.slots;
    let __VLS_200;
    /** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
    elCard;
    // @ts-ignore
    const __VLS_201 = __VLS_asFunctionalComponent1(__VLS_200, new __VLS_200({
        shadow: "never",
        ...{ class: "cardBg" },
    }));
    const __VLS_202 = __VLS_201({
        shadow: "never",
        ...{ class: "cardBg" },
    }, ...__VLS_functionalComponentArgsRest(__VLS_201));
    /** @type {__VLS_StyleScopedClasses['cardBg']} */ ;
    const { default: __VLS_205 } = __VLS_203.slots;
    for (const [item, index] of __VLS_vFor((__VLS_ctx.state.selectProductAttrPics))) {
        __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
            key: ('productAttrPic' + index),
        });
        __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
        (item.name);
        const __VLS_206 = SingleUpload || SingleUpload;
        // @ts-ignore
        const __VLS_207 = __VLS_asFunctionalComponent1(__VLS_206, new __VLS_206({
            modelValue: (item.pic),
            ...{ style: {} },
        }));
        const __VLS_208 = __VLS_207({
            modelValue: (item.pic),
            ...{ style: {} },
        }, ...__VLS_functionalComponentArgsRest(__VLS_207));
        // @ts-ignore
        [state, hasAttrPic,];
    }
    // @ts-ignore
    [];
    var __VLS_203;
    // @ts-ignore
    [];
    var __VLS_197;
}
let __VLS_211;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_212 = __VLS_asFunctionalComponent1(__VLS_211, new __VLS_211({
    label: "商品参数：",
}));
const __VLS_213 = __VLS_212({
    label: "商品参数：",
}, ...__VLS_functionalComponentArgsRest(__VLS_212));
const { default: __VLS_216 } = __VLS_214.slots;
let __VLS_217;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_218 = __VLS_asFunctionalComponent1(__VLS_217, new __VLS_217({
    shadow: "never",
    ...{ class: "cardBg" },
}));
const __VLS_219 = __VLS_218({
    shadow: "never",
    ...{ class: "cardBg" },
}, ...__VLS_functionalComponentArgsRest(__VLS_218));
/** @type {__VLS_StyleScopedClasses['cardBg']} */ ;
const { default: __VLS_222 } = __VLS_220.slots;
for (const [item, index] of __VLS_vFor((__VLS_ctx.state.selectProductParam))) {
    __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
        ...{ class: ({ littleMarginTop: index !== 0 }) },
        key: ('productParam' + index),
    });
    /** @type {__VLS_StyleScopedClasses['littleMarginTop']} */ ;
    __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
        ...{ class: "paramInputLabel" },
    });
    /** @type {__VLS_StyleScopedClasses['paramInputLabel']} */ ;
    (item.name);
    if (item.inputType === 1) {
        let __VLS_223;
        /** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
        elSelect;
        // @ts-ignore
        const __VLS_224 = __VLS_asFunctionalComponent1(__VLS_223, new __VLS_223({
            ...{ class: "paramInput" },
            modelValue: (__VLS_ctx.state.selectProductParam[index].value),
        }));
        const __VLS_225 = __VLS_224({
            ...{ class: "paramInput" },
            modelValue: (__VLS_ctx.state.selectProductParam[index].value),
        }, ...__VLS_functionalComponentArgsRest(__VLS_224));
        /** @type {__VLS_StyleScopedClasses['paramInput']} */ ;
        const { default: __VLS_228 } = __VLS_226.slots;
        for (const [item2] of __VLS_vFor((__VLS_ctx.getParamInputList(item.inputList)))) {
            let __VLS_229;
            /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
            elOption;
            // @ts-ignore
            const __VLS_230 = __VLS_asFunctionalComponent1(__VLS_229, new __VLS_229({
                key: (item2),
                label: (item2),
                value: (item2),
            }));
            const __VLS_231 = __VLS_230({
                key: (item2),
                label: (item2),
                value: (item2),
            }, ...__VLS_functionalComponentArgsRest(__VLS_230));
            // @ts-ignore
            [state, state, getParamInputList,];
        }
        // @ts-ignore
        [];
        var __VLS_226;
    }
    else {
        let __VLS_234;
        /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
        elInput;
        // @ts-ignore
        const __VLS_235 = __VLS_asFunctionalComponent1(__VLS_234, new __VLS_234({
            ...{ class: "paramInput" },
            modelValue: (__VLS_ctx.state.selectProductParam[index].value),
        }));
        const __VLS_236 = __VLS_235({
            ...{ class: "paramInput" },
            modelValue: (__VLS_ctx.state.selectProductParam[index].value),
        }, ...__VLS_functionalComponentArgsRest(__VLS_235));
        /** @type {__VLS_StyleScopedClasses['paramInput']} */ ;
    }
    // @ts-ignore
    [state,];
}
// @ts-ignore
[];
var __VLS_220;
// @ts-ignore
[];
var __VLS_214;
let __VLS_239;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_240 = __VLS_asFunctionalComponent1(__VLS_239, new __VLS_239({
    label: "商品相册：",
}));
const __VLS_241 = __VLS_240({
    label: "商品相册：",
}, ...__VLS_functionalComponentArgsRest(__VLS_240));
const { default: __VLS_244 } = __VLS_242.slots;
const __VLS_245 = MultiUpload || MultiUpload;
// @ts-ignore
const __VLS_246 = __VLS_asFunctionalComponent1(__VLS_245, new __VLS_245({
    modelValue: (__VLS_ctx.selectProductPics),
}));
const __VLS_247 = __VLS_246({
    modelValue: (__VLS_ctx.selectProductPics),
}, ...__VLS_functionalComponentArgsRest(__VLS_246));
// @ts-ignore
[selectProductPics,];
var __VLS_242;
let __VLS_250;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_251 = __VLS_asFunctionalComponent1(__VLS_250, new __VLS_250({
    label: "商品详情：",
}));
const __VLS_252 = __VLS_251({
    label: "商品详情：",
}, ...__VLS_functionalComponentArgsRest(__VLS_251));
const { default: __VLS_255 } = __VLS_253.slots;
let __VLS_256;
/** @ts-ignore @type { | typeof __VLS_components.elTabs | typeof __VLS_components.ElTabs | typeof __VLS_components['el-tabs'] | typeof __VLS_components.elTabs | typeof __VLS_components.ElTabs | typeof __VLS_components['el-tabs']} */
elTabs;
// @ts-ignore
const __VLS_257 = __VLS_asFunctionalComponent1(__VLS_256, new __VLS_256({
    modelValue: (__VLS_ctx.state.activeHtmlName),
    type: "card",
    ...{ style: {} },
}));
const __VLS_258 = __VLS_257({
    modelValue: (__VLS_ctx.state.activeHtmlName),
    type: "card",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_257));
const { default: __VLS_261 } = __VLS_259.slots;
let __VLS_262;
/** @ts-ignore @type { | typeof __VLS_components.elTabPane | typeof __VLS_components.ElTabPane | typeof __VLS_components['el-tab-pane'] | typeof __VLS_components.elTabPane | typeof __VLS_components.ElTabPane | typeof __VLS_components['el-tab-pane']} */
elTabPane;
// @ts-ignore
const __VLS_263 = __VLS_asFunctionalComponent1(__VLS_262, new __VLS_262({
    label: "电脑端详情",
    name: "pc",
}));
const __VLS_264 = __VLS_263({
    label: "电脑端详情",
    name: "pc",
}, ...__VLS_functionalComponentArgsRest(__VLS_263));
const { default: __VLS_267 } = __VLS_265.slots;
const __VLS_268 = Tinymce;
// @ts-ignore
const __VLS_269 = __VLS_asFunctionalComponent1(__VLS_268, new __VLS_268({
    height: (720),
    modelValue: (__VLS_ctx.compProductParam.detailHtml),
}));
const __VLS_270 = __VLS_269({
    height: (720),
    modelValue: (__VLS_ctx.compProductParam.detailHtml),
}, ...__VLS_functionalComponentArgsRest(__VLS_269));
// @ts-ignore
[compProductParam, state,];
var __VLS_265;
let __VLS_273;
/** @ts-ignore @type { | typeof __VLS_components.elTabPane | typeof __VLS_components.ElTabPane | typeof __VLS_components['el-tab-pane'] | typeof __VLS_components.elTabPane | typeof __VLS_components.ElTabPane | typeof __VLS_components['el-tab-pane']} */
elTabPane;
// @ts-ignore
const __VLS_274 = __VLS_asFunctionalComponent1(__VLS_273, new __VLS_273({
    label: "移动端详情",
    name: "mobile",
}));
const __VLS_275 = __VLS_274({
    label: "移动端详情",
    name: "mobile",
}, ...__VLS_functionalComponentArgsRest(__VLS_274));
const { default: __VLS_278 } = __VLS_276.slots;
const __VLS_279 = Tinymce;
// @ts-ignore
const __VLS_280 = __VLS_asFunctionalComponent1(__VLS_279, new __VLS_279({
    height: (720),
    modelValue: (__VLS_ctx.compProductParam.detailMobileHtml),
}));
const __VLS_281 = __VLS_280({
    height: (720),
    modelValue: (__VLS_ctx.compProductParam.detailMobileHtml),
}, ...__VLS_functionalComponentArgsRest(__VLS_280));
// @ts-ignore
[compProductParam,];
var __VLS_276;
// @ts-ignore
[];
var __VLS_259;
// @ts-ignore
[];
var __VLS_253;
let __VLS_284;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_285 = __VLS_asFunctionalComponent1(__VLS_284, new __VLS_284({}));
const __VLS_286 = __VLS_285({}, ...__VLS_functionalComponentArgsRest(__VLS_285));
const { default: __VLS_289 } = __VLS_287.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_290;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_291 = __VLS_asFunctionalComponent1(__VLS_290, new __VLS_290({
    ...{ 'onClick': {} },
}));
const __VLS_292 = __VLS_291({
    ...{ 'onClick': {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_291));
let __VLS_295;
const __VLS_296 = ({ click: {} },
    { onClick: (__VLS_ctx.handlePrev) });
const { default: __VLS_297 } = __VLS_293.slots;
// @ts-ignore
[handlePrev,];
var __VLS_293;
var __VLS_294;
let __VLS_298;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_299 = __VLS_asFunctionalComponent1(__VLS_298, new __VLS_298({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_300 = __VLS_299({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_299));
let __VLS_303;
const __VLS_304 = ({ click: {} },
    { onClick: (__VLS_ctx.handleNext) });
const { default: __VLS_305 } = __VLS_301.slots;
// @ts-ignore
[handleNext,];
var __VLS_301;
var __VLS_302;
// @ts-ignore
[];
var __VLS_287;
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
