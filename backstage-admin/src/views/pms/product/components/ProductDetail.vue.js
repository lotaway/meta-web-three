/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted, provide, } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import ProductInfoDetail from './ProductInfoDetail.vue';
import ProductSaleDetail from './ProductSaleDetail.vue';
import ProductAttrDetail from './ProductAttrDetail.vue';
import ProductRelationDetail from './ProductRelationDetail.vue';
import { productCreateAPI, getPruductUpdateInfoAPI, productUpdateByIdAPI } from '@/apis/product';
// 获取路由对象
const route = useRoute();
const router = useRouter();
// 定义属性
const props = defineProps({
    isEdit: {
        type: Boolean,
        default: false
    }
});
// 默认商品参数
const defaultProductParam = {
    albumPics: '',
    brandName: '',
    deleteStatus: 0,
    description: '',
    detailDesc: '',
    detailHtml: '',
    detailMobileHtml: '',
    detailTitle: '',
    feightTemplateId: 0,
    flashPromotionCount: 0,
    flashPromotionId: 0,
    flashPromotionPrice: 0,
    flashPromotionSort: 0,
    giftPoint: 0,
    giftGrowth: 0,
    keywords: '',
    lowStock: 0,
    name: '',
    newStatus: 0,
    note: '',
    originalPrice: 0,
    pic: '',
    memberPriceList: [],
    productFullReductionList: [{ fullPrice: 0, reducePrice: 0 }],
    productLadderList: [{ count: 0, discount: 0, price: 0 }],
    previewStatus: 0,
    price: 0,
    productAttributeValueList: [],
    skuStockList: [],
    subjectProductRelationList: [],
    prefrenceAreaProductRelationList: [],
    productCategoryName: '',
    productSn: '',
    promotionEndTime: '',
    promotionPerLimit: 0,
    promotionStartTime: '',
    promotionType: 0,
    publishStatus: 0,
    recommandStatus: 0,
    sale: 0,
    serviceIds: '',
    sort: 0,
    stock: 0,
    subTitle: '',
    unit: '',
    usePointLimit: 0,
    verifyStatus: 0,
    weight: 0
};
// el-steps的激活状态
const active = ref(0);
// 商品信息/促销/属性/关联组件显示状态
const showStatus = ref([true, false, false, false]);
// 当前商品参数
const productParam = ref(Object.assign({}, defaultProductParam));
// 实现数据跨层传递
provide('product-key', productParam);
// 组件挂载后执行
onMounted(async () => {
    if (props.isEdit) {
        const res = await getPruductUpdateInfoAPI(Number(route.query.id));
        productParam.value = res.data;
    }
});
// 隐藏所有商品编辑组件
const hideAll = () => {
    // 将所有状态设置为false
    showStatus.value.fill(false);
};
// 上一步
const prevStep = () => {
    if (active.value > 0 && active.value < showStatus.value.length) {
        active.value--;
        hideAll();
        showStatus.value[active.value] = true;
    }
};
// 下一步
const nextStep = () => {
    if (active.value < showStatus.value.length - 1) {
        active.value++;
        hideAll();
        showStatus.value[active.value] = true;
    }
};
// 结束步骤，提交数据
const finishCommit = async (isEdit) => {
    await ElMessageBox.confirm('是否要提交该商品？', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    if (isEdit) {
        await productUpdateByIdAPI(Number(route.query.id), productParam.value);
        ElMessage({
            type: 'success',
            message: '提交成功',
            duration: 1000
        });
        router.back();
    }
    else {
        await productCreateAPI(productParam.value);
        ElMessage({
            type: 'success',
            message: '提交成功',
            duration: 1000
        });
        location.reload();
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
/** @ts-ignore @type { | typeof __VLS_components.elSteps | typeof __VLS_components.ElSteps | typeof __VLS_components['el-steps'] | typeof __VLS_components.elSteps | typeof __VLS_components.ElSteps | typeof __VLS_components['el-steps']} */
elSteps;
// @ts-ignore
const __VLS_8 = __VLS_asFunctionalComponent1(__VLS_7, new __VLS_7({
    active: (__VLS_ctx.active),
    finishStatus: "success",
    alignCenter: true,
}));
const __VLS_9 = __VLS_8({
    active: (__VLS_ctx.active),
    finishStatus: "success",
    alignCenter: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
const { default: __VLS_12 } = __VLS_10.slots;
let __VLS_13;
/** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
elStep;
// @ts-ignore
const __VLS_14 = __VLS_asFunctionalComponent1(__VLS_13, new __VLS_13({
    title: "填写商品信息",
}));
const __VLS_15 = __VLS_14({
    title: "填写商品信息",
}, ...__VLS_functionalComponentArgsRest(__VLS_14));
let __VLS_18;
/** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
elStep;
// @ts-ignore
const __VLS_19 = __VLS_asFunctionalComponent1(__VLS_18, new __VLS_18({
    title: "填写商品促销",
}));
const __VLS_20 = __VLS_19({
    title: "填写商品促销",
}, ...__VLS_functionalComponentArgsRest(__VLS_19));
let __VLS_23;
/** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
elStep;
// @ts-ignore
const __VLS_24 = __VLS_asFunctionalComponent1(__VLS_23, new __VLS_23({
    title: "填写商品属性",
}));
const __VLS_25 = __VLS_24({
    title: "填写商品属性",
}, ...__VLS_functionalComponentArgsRest(__VLS_24));
let __VLS_28;
/** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
elStep;
// @ts-ignore
const __VLS_29 = __VLS_asFunctionalComponent1(__VLS_28, new __VLS_28({
    title: "选择商品关联",
}));
const __VLS_30 = __VLS_29({
    title: "选择商品关联",
}, ...__VLS_functionalComponentArgsRest(__VLS_29));
// @ts-ignore
[active,];
var __VLS_10;
const __VLS_33 = ProductInfoDetail || ProductInfoDetail;
// @ts-ignore
const __VLS_34 = __VLS_asFunctionalComponent1(__VLS_33, new __VLS_33({
    ...{ 'onNextStep': {} },
    isEdit: (__VLS_ctx.isEdit),
}));
const __VLS_35 = __VLS_34({
    ...{ 'onNextStep': {} },
    isEdit: (__VLS_ctx.isEdit),
}, ...__VLS_functionalComponentArgsRest(__VLS_34));
let __VLS_38;
const __VLS_39 = ({ nextStep: {} },
    { onNextStep: (__VLS_ctx.nextStep) });
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.showStatus[0]) }, null, null);
var __VLS_36;
var __VLS_37;
const __VLS_40 = ProductSaleDetail || ProductSaleDetail;
// @ts-ignore
const __VLS_41 = __VLS_asFunctionalComponent1(__VLS_40, new __VLS_40({
    ...{ 'onNextStep': {} },
    ...{ 'onPrevStep': {} },
    isEdit: (__VLS_ctx.isEdit),
}));
const __VLS_42 = __VLS_41({
    ...{ 'onNextStep': {} },
    ...{ 'onPrevStep': {} },
    isEdit: (__VLS_ctx.isEdit),
}, ...__VLS_functionalComponentArgsRest(__VLS_41));
let __VLS_45;
const __VLS_46 = ({ nextStep: {} },
    { onNextStep: (__VLS_ctx.nextStep) });
const __VLS_47 = ({ prevStep: {} },
    { onPrevStep: (__VLS_ctx.prevStep) });
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.showStatus[1]) }, null, null);
var __VLS_43;
var __VLS_44;
const __VLS_48 = ProductAttrDetail || ProductAttrDetail;
// @ts-ignore
const __VLS_49 = __VLS_asFunctionalComponent1(__VLS_48, new __VLS_48({
    ...{ 'onNextStep': {} },
    ...{ 'onPrevStep': {} },
    modelValue: (__VLS_ctx.productParam),
    isEdit: (__VLS_ctx.isEdit),
}));
const __VLS_50 = __VLS_49({
    ...{ 'onNextStep': {} },
    ...{ 'onPrevStep': {} },
    modelValue: (__VLS_ctx.productParam),
    isEdit: (__VLS_ctx.isEdit),
}, ...__VLS_functionalComponentArgsRest(__VLS_49));
let __VLS_53;
const __VLS_54 = ({ nextStep: {} },
    { onNextStep: (__VLS_ctx.nextStep) });
const __VLS_55 = ({ prevStep: {} },
    { onPrevStep: (__VLS_ctx.prevStep) });
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.showStatus[2]) }, null, null);
var __VLS_51;
var __VLS_52;
const __VLS_56 = ProductRelationDetail || ProductRelationDetail;
// @ts-ignore
const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
    ...{ 'onPrevStep': {} },
    ...{ 'onFinishCommit': {} },
    modelValue: (__VLS_ctx.productParam),
    isEdit: (__VLS_ctx.isEdit),
}));
const __VLS_58 = __VLS_57({
    ...{ 'onPrevStep': {} },
    ...{ 'onFinishCommit': {} },
    modelValue: (__VLS_ctx.productParam),
    isEdit: (__VLS_ctx.isEdit),
}, ...__VLS_functionalComponentArgsRest(__VLS_57));
let __VLS_61;
const __VLS_62 = ({ prevStep: {} },
    { onPrevStep: (__VLS_ctx.prevStep) });
const __VLS_63 = ({ finishCommit: {} },
    { onFinishCommit: (__VLS_ctx.finishCommit) });
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.showStatus[3]) }, null, null);
var __VLS_59;
var __VLS_60;
// @ts-ignore
[isEdit, isEdit, isEdit, isEdit, nextStep, nextStep, nextStep, showStatus, showStatus, showStatus, showStatus, prevStep, prevStep, prevStep, productParam, productParam, finishCommit,];
var __VLS_3;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({
    props: {
        isEdit: {
            type: Boolean,
            default: false
        }
    },
});
export default {};
