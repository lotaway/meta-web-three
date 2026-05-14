/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, reactive, onMounted } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { couponCreateAPI, getCouponByIdAPI, couponUpdateByIdAPI } from '@/apis/coupon';
import { getProductListAPI } from '@/apis/product';
import { getProductCategoryListWithChildrenAPI } from '@/apis/productCate';
import { useRoute, useRouter } from 'vue-router';
import { couponPlatforms, couponTypes } from '@/utils/constant';
// 获取路由
const route = useRoute();
const router = useRouter();
// 定义属性
const props = defineProps({
    isEdit: {
        type: Boolean,
        default: false
    }
});
// 默认优惠券，添加时使用
const defaultCoupon = {
    type: 0,
    name: '',
    platform: 0,
    amount: 0,
    perLimit: 1,
    useType: 0,
    productRelationList: [],
    productCategoryRelationList: []
};
// 当前操作的优惠券对象
const coupon = ref(Object.assign({}, defaultCoupon));
// 选择使用类型为指定分类时的商品分类信息
const productCateOptions = ref([]);
// 优惠券类型为指定分类时所选商品分类ID([父级,子级])
const selectProductCate = ref([]);
// 获取分类列表
const getProductCateList = async () => {
    try {
        const res = await getProductCategoryListWithChildrenAPI();
        const list = res.data;
        productCateOptions.value = list.map(item => ({
            label: item.name,
            value: item.id,
            children: item.children?.map(it => ({ label: it.name, value: it.id }))
        }));
    }
    catch (error) {
        console.error('获取商品分类列表失败:', error);
    }
};
// 初始化数据
onMounted(async () => {
    if (props.isEdit) {
        const response = await getCouponByIdAPI(Number(route.query.id));
        coupon.value = response.data;
    }
    await getProductCateList();
});
// 优惠券类型为指定商品时表格数据加载状态
const selectProductLoading = ref(false);
// 优惠券类型为指定商品时表格中提供的商品列表选项
const selectProductOptions = ref([]);
// 优惠券类型为指定商品时当前选中的关联商品ID
const selectProduct = ref();
// 表单引用
const couponFrom = ref();
// 表单验证规则
const rules = reactive({
    name: [
        { required: true, message: '请输入优惠券名称', trigger: 'blur' },
        { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
    ],
    publishCount: [
        { type: 'number', required: true, message: '只能输入正整数', trigger: 'blur' }
    ],
    amount: [
        { type: 'number', required: true, message: '面值只能是数值，0.01-10000，限2位小数', trigger: 'blur' }
    ],
    minPoint: [
        { type: 'number', required: true, message: '只能输入正整数', trigger: 'blur' }
    ]
});
// 提交表单
const onSubmit = async () => {
    if (!couponFrom.value)
        return;
    const isValid = await couponFrom.value.validate().catch(() => false);
    if (isValid) {
        await ElMessageBox.confirm('是否提交数据', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        if (props.isEdit) {
            try {
                await couponUpdateByIdAPI(Number(route.query.id), coupon.value);
                couponFrom.value.resetFields();
                ElMessage({
                    message: '修改成功',
                    type: 'success',
                    duration: 1000
                });
                router.back();
            }
            catch (error) {
                console.error('更新优惠券失败:', error);
            }
        }
        else {
            try {
                await couponCreateAPI(coupon.value);
                couponFrom.value.resetFields();
                ElMessage({
                    message: '提交成功',
                    type: 'success',
                    duration: 1000
                });
                router.back();
            }
            catch (error) {
                console.error('创建优惠券失败:', error);
            }
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
    if (couponFrom.value) {
        couponFrom.value.resetFields();
    }
    coupon.value = Object.assign({}, defaultCoupon);
};
// 搜索商品方法
const searchProductMethod = async (query) => {
    if (query) {
        selectProductLoading.value = true;
        try {
            const res = await getProductListAPI({ pageNum: 1, pageSize: 10, keyword: query });
            selectProductLoading.value = false;
            const productList = res.data.list;
            selectProductOptions.value = productList.map(item => ({
                productId: item.id,
                productName: item.name,
                productSn: item.productSn
            }));
        }
        catch (error) {
            selectProductLoading.value = false;
            console.error('搜索商品失败:', error);
        }
    }
    else {
        selectProductOptions.value = [];
    }
};
// 添加商品关联
const handleAddProductRelation = () => {
    if (!selectProduct.value) {
        ElMessage({
            message: '请先选择商品',
            type: 'warning'
        });
        return;
    }
    coupon.value.productRelationList.push(getProductById(selectProduct.value));
    selectProduct.value = undefined;
};
// 删除商品关联
const handleDeleteProductRelation = (index) => {
    coupon.value.productRelationList.splice(index, 1);
};
// 添加商品分类关联
const handleAddProductCategoryRelation = () => {
    if (selectProductCate.value.length <= 0) {
        ElMessage({
            message: '请先选择商品分类',
            type: 'warning'
        });
        return;
    }
    coupon.value.productCategoryRelationList.push(getProductCateByIds(selectProductCate.value));
    selectProductCate.value = [];
};
// 删除商品分类关联
const handleDeleteProductCateRelation = (index) => {
    coupon.value.productCategoryRelationList.splice(index, 1);
};
// 根据ID获取关联商品
const getProductById = (id) => {
    return selectProductOptions.value.find(item => item.productId === id);
};
// 根据IDs获取商品分类
const getProductCateByIds = (ids) => {
    const findParentCate = productCateOptions.value.find(item => item.value === ids[0]);
    const findCate = findParentCate?.children?.find(item => item.value === ids[1]);
    return { productCategoryId: ids[1], productCategoryName: findCate?.label, parentCategoryName: findParentCate?.label };
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
    model: (__VLS_ctx.coupon),
    rules: (__VLS_ctx.rules),
    ref: "couponFrom",
    labelWidth: "150px",
}));
const __VLS_9 = __VLS_8({
    model: (__VLS_ctx.coupon),
    rules: (__VLS_ctx.rules),
    ref: "couponFrom",
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
var __VLS_12 = {};
const { default: __VLS_14 } = __VLS_10.slots;
let __VLS_15;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_16 = __VLS_asFunctionalComponent1(__VLS_15, new __VLS_15({
    label: "优惠券类型：",
}));
const __VLS_17 = __VLS_16({
    label: "优惠券类型：",
}, ...__VLS_functionalComponentArgsRest(__VLS_16));
const { default: __VLS_20 } = __VLS_18.slots;
let __VLS_21;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_22 = __VLS_asFunctionalComponent1(__VLS_21, new __VLS_21({
    modelValue: (__VLS_ctx.coupon.type),
}));
const __VLS_23 = __VLS_22({
    modelValue: (__VLS_ctx.coupon.type),
}, ...__VLS_functionalComponentArgsRest(__VLS_22));
const { default: __VLS_26 } = __VLS_24.slots;
for (const [type] of __VLS_vFor((__VLS_ctx.couponTypes))) {
    let __VLS_27;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_28 = __VLS_asFunctionalComponent1(__VLS_27, new __VLS_27({
        key: (type.value),
        label: (type.label),
        value: (type.value),
    }));
    const __VLS_29 = __VLS_28({
        key: (type.value),
        label: (type.label),
        value: (type.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_28));
    // @ts-ignore
    [coupon, coupon, rules, couponTypes,];
}
// @ts-ignore
[];
var __VLS_24;
// @ts-ignore
[];
var __VLS_18;
let __VLS_32;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_33 = __VLS_asFunctionalComponent1(__VLS_32, new __VLS_32({
    label: "优惠券名称：",
    prop: "name",
}));
const __VLS_34 = __VLS_33({
    label: "优惠券名称：",
    prop: "name",
}, ...__VLS_functionalComponentArgsRest(__VLS_33));
const { default: __VLS_37 } = __VLS_35.slots;
let __VLS_38;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_39 = __VLS_asFunctionalComponent1(__VLS_38, new __VLS_38({
    modelValue: (__VLS_ctx.coupon.name),
    ...{ class: "input-width" },
}));
const __VLS_40 = __VLS_39({
    modelValue: (__VLS_ctx.coupon.name),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_39));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[coupon,];
var __VLS_35;
let __VLS_43;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_44 = __VLS_asFunctionalComponent1(__VLS_43, new __VLS_43({
    label: "适用平台：",
}));
const __VLS_45 = __VLS_44({
    label: "适用平台：",
}, ...__VLS_functionalComponentArgsRest(__VLS_44));
const { default: __VLS_48 } = __VLS_46.slots;
let __VLS_49;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_50 = __VLS_asFunctionalComponent1(__VLS_49, new __VLS_49({
    modelValue: (__VLS_ctx.coupon.platform),
}));
const __VLS_51 = __VLS_50({
    modelValue: (__VLS_ctx.coupon.platform),
}, ...__VLS_functionalComponentArgsRest(__VLS_50));
const { default: __VLS_54 } = __VLS_52.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.couponPlatforms))) {
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
    [coupon, couponPlatforms,];
}
// @ts-ignore
[];
var __VLS_52;
// @ts-ignore
[];
var __VLS_46;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    label: "总发行量：",
    prop: "publishCount",
}));
const __VLS_62 = __VLS_61({
    label: "总发行量：",
    prop: "publishCount",
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
const { default: __VLS_65 } = __VLS_63.slots;
let __VLS_66;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_67 = __VLS_asFunctionalComponent1(__VLS_66, new __VLS_66({
    modelValue: (__VLS_ctx.coupon.publishCount),
    modelModifiers: { number: true, },
    placeholder: "只能输入正整数",
    ...{ class: "input-width" },
}));
const __VLS_68 = __VLS_67({
    modelValue: (__VLS_ctx.coupon.publishCount),
    modelModifiers: { number: true, },
    placeholder: "只能输入正整数",
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_67));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[coupon,];
var __VLS_63;
let __VLS_71;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_72 = __VLS_asFunctionalComponent1(__VLS_71, new __VLS_71({
    label: "面额：",
    prop: "amount",
}));
const __VLS_73 = __VLS_72({
    label: "面额：",
    prop: "amount",
}, ...__VLS_functionalComponentArgsRest(__VLS_72));
const { default: __VLS_76 } = __VLS_74.slots;
let __VLS_77;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_78 = __VLS_asFunctionalComponent1(__VLS_77, new __VLS_77({
    modelValue: (__VLS_ctx.coupon.amount),
    modelModifiers: { number: true, },
    placeholder: "面值只能是数值，限2位小数",
    ...{ class: "input-width" },
}));
const __VLS_79 = __VLS_78({
    modelValue: (__VLS_ctx.coupon.amount),
    modelModifiers: { number: true, },
    placeholder: "面值只能是数值，限2位小数",
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_78));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_82 } = __VLS_80.slots;
{
    const { append: __VLS_83 } = __VLS_80.slots;
    // @ts-ignore
    [coupon,];
}
// @ts-ignore
[];
var __VLS_80;
// @ts-ignore
[];
var __VLS_74;
let __VLS_84;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_85 = __VLS_asFunctionalComponent1(__VLS_84, new __VLS_84({
    label: "每人限领：",
}));
const __VLS_86 = __VLS_85({
    label: "每人限领：",
}, ...__VLS_functionalComponentArgsRest(__VLS_85));
const { default: __VLS_89 } = __VLS_87.slots;
let __VLS_90;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_91 = __VLS_asFunctionalComponent1(__VLS_90, new __VLS_90({
    modelValue: (__VLS_ctx.coupon.perLimit),
    placeholder: "只能输入正整数",
    ...{ class: "input-width" },
}));
const __VLS_92 = __VLS_91({
    modelValue: (__VLS_ctx.coupon.perLimit),
    placeholder: "只能输入正整数",
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_91));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_95 } = __VLS_93.slots;
{
    const { append: __VLS_96 } = __VLS_93.slots;
    // @ts-ignore
    [coupon,];
}
// @ts-ignore
[];
var __VLS_93;
// @ts-ignore
[];
var __VLS_87;
let __VLS_97;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_98 = __VLS_asFunctionalComponent1(__VLS_97, new __VLS_97({
    label: "使用门槛：",
    prop: "minPoint",
}));
const __VLS_99 = __VLS_98({
    label: "使用门槛：",
    prop: "minPoint",
}, ...__VLS_functionalComponentArgsRest(__VLS_98));
const { default: __VLS_102 } = __VLS_100.slots;
let __VLS_103;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_104 = __VLS_asFunctionalComponent1(__VLS_103, new __VLS_103({
    modelValue: (__VLS_ctx.coupon.minPoint),
    modelModifiers: { number: true, },
    placeholder: "只能输入正整数",
    ...{ class: "input-width" },
}));
const __VLS_105 = __VLS_104({
    modelValue: (__VLS_ctx.coupon.minPoint),
    modelModifiers: { number: true, },
    placeholder: "只能输入正整数",
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_104));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_108 } = __VLS_106.slots;
{
    const { prepend: __VLS_109 } = __VLS_106.slots;
    // @ts-ignore
    [coupon,];
}
{
    const { append: __VLS_110 } = __VLS_106.slots;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_106;
// @ts-ignore
[];
var __VLS_100;
let __VLS_111;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_112 = __VLS_asFunctionalComponent1(__VLS_111, new __VLS_111({
    label: "领取日期：",
    prop: "enableTime",
}));
const __VLS_113 = __VLS_112({
    label: "领取日期：",
    prop: "enableTime",
}, ...__VLS_functionalComponentArgsRest(__VLS_112));
const { default: __VLS_116 } = __VLS_114.slots;
let __VLS_117;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_118 = __VLS_asFunctionalComponent1(__VLS_117, new __VLS_117({
    type: "date",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.coupon.enableTime),
    ...{ class: "input-width" },
}));
const __VLS_119 = __VLS_118({
    type: "date",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.coupon.enableTime),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_118));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[coupon,];
var __VLS_114;
let __VLS_122;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_123 = __VLS_asFunctionalComponent1(__VLS_122, new __VLS_122({
    label: "有效期：",
}));
const __VLS_124 = __VLS_123({
    label: "有效期：",
}, ...__VLS_functionalComponentArgsRest(__VLS_123));
const { default: __VLS_127 } = __VLS_125.slots;
let __VLS_128;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_129 = __VLS_asFunctionalComponent1(__VLS_128, new __VLS_128({
    type: "date",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.coupon.startTime),
    ...{ style: {} },
}));
const __VLS_130 = __VLS_129({
    type: "date",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.coupon.startTime),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_129));
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
});
let __VLS_133;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_134 = __VLS_asFunctionalComponent1(__VLS_133, new __VLS_133({
    type: "date",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.coupon.endTime),
    ...{ style: {} },
}));
const __VLS_135 = __VLS_134({
    type: "date",
    placeholder: "选择日期",
    modelValue: (__VLS_ctx.coupon.endTime),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_134));
// @ts-ignore
[coupon, coupon,];
var __VLS_125;
let __VLS_138;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_139 = __VLS_asFunctionalComponent1(__VLS_138, new __VLS_138({
    label: "可使用商品：",
}));
const __VLS_140 = __VLS_139({
    label: "可使用商品：",
}, ...__VLS_functionalComponentArgsRest(__VLS_139));
const { default: __VLS_143 } = __VLS_141.slots;
let __VLS_144;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_145 = __VLS_asFunctionalComponent1(__VLS_144, new __VLS_144({
    modelValue: (__VLS_ctx.coupon.useType),
}));
const __VLS_146 = __VLS_145({
    modelValue: (__VLS_ctx.coupon.useType),
}, ...__VLS_functionalComponentArgsRest(__VLS_145));
const { default: __VLS_149 } = __VLS_147.slots;
let __VLS_150;
/** @ts-ignore @type { | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button'] | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button']} */
elRadioButton;
// @ts-ignore
const __VLS_151 = __VLS_asFunctionalComponent1(__VLS_150, new __VLS_150({
    label: (0),
}));
const __VLS_152 = __VLS_151({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_151));
const { default: __VLS_155 } = __VLS_153.slots;
// @ts-ignore
[coupon,];
var __VLS_153;
let __VLS_156;
/** @ts-ignore @type { | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button'] | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button']} */
elRadioButton;
// @ts-ignore
const __VLS_157 = __VLS_asFunctionalComponent1(__VLS_156, new __VLS_156({
    label: (1),
}));
const __VLS_158 = __VLS_157({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_157));
const { default: __VLS_161 } = __VLS_159.slots;
// @ts-ignore
[];
var __VLS_159;
let __VLS_162;
/** @ts-ignore @type { | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button'] | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button']} */
elRadioButton;
// @ts-ignore
const __VLS_163 = __VLS_asFunctionalComponent1(__VLS_162, new __VLS_162({
    label: (2),
}));
const __VLS_164 = __VLS_163({
    label: (2),
}, ...__VLS_functionalComponentArgsRest(__VLS_163));
const { default: __VLS_167 } = __VLS_165.slots;
// @ts-ignore
[];
var __VLS_165;
// @ts-ignore
[];
var __VLS_147;
// @ts-ignore
[];
var __VLS_141;
let __VLS_168;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_169 = __VLS_asFunctionalComponent1(__VLS_168, new __VLS_168({}));
const __VLS_170 = __VLS_169({}, ...__VLS_functionalComponentArgsRest(__VLS_169));
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.coupon.useType === 1) }, null, null);
const { default: __VLS_173 } = __VLS_171.slots;
let __VLS_174;
/** @ts-ignore @type { | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader'] | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader']} */
elCascader;
// @ts-ignore
const __VLS_175 = __VLS_asFunctionalComponent1(__VLS_174, new __VLS_174({
    clearable: true,
    placeholder: "请选择分类名称",
    modelValue: (__VLS_ctx.selectProductCate),
    options: (__VLS_ctx.productCateOptions),
}));
const __VLS_176 = __VLS_175({
    clearable: true,
    placeholder: "请选择分类名称",
    modelValue: (__VLS_ctx.selectProductCate),
    options: (__VLS_ctx.productCateOptions),
}, ...__VLS_functionalComponentArgsRest(__VLS_175));
let __VLS_179;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_180 = __VLS_asFunctionalComponent1(__VLS_179, new __VLS_179({
    ...{ 'onClick': {} },
}));
const __VLS_181 = __VLS_180({
    ...{ 'onClick': {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_180));
let __VLS_184;
const __VLS_185 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleAddProductCategoryRelation();
            // @ts-ignore
            [coupon, selectProductCate, productCateOptions, handleAddProductCategoryRelation,];
        } });
const { default: __VLS_186 } = __VLS_182.slots;
// @ts-ignore
[];
var __VLS_182;
var __VLS_183;
let __VLS_187;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_188 = __VLS_asFunctionalComponent1(__VLS_187, new __VLS_187({
    ref: "productCateRelationTable",
    data: (__VLS_ctx.coupon.productCategoryRelationList),
    ...{ style: {} },
    border: true,
}));
const __VLS_189 = __VLS_188({
    ref: "productCateRelationTable",
    data: (__VLS_ctx.coupon.productCategoryRelationList),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_188));
var __VLS_192 = {};
const { default: __VLS_194 } = __VLS_190.slots;
let __VLS_195;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_196 = __VLS_asFunctionalComponent1(__VLS_195, new __VLS_195({
    label: "分类名称",
    align: "center",
}));
const __VLS_197 = __VLS_196({
    label: "分类名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_196));
const { default: __VLS_200 } = __VLS_198.slots;
{
    const { default: __VLS_201 } = __VLS_198.slots;
    const [scope] = __VLS_vSlot(__VLS_201);
    (scope.row.parentCategoryName);
    (scope.row.productCategoryName);
    // @ts-ignore
    [coupon,];
}
// @ts-ignore
[];
var __VLS_198;
let __VLS_202;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_203 = __VLS_asFunctionalComponent1(__VLS_202, new __VLS_202({
    label: "操作",
    align: "center",
    width: "100",
}));
const __VLS_204 = __VLS_203({
    label: "操作",
    align: "center",
    width: "100",
}, ...__VLS_functionalComponentArgsRest(__VLS_203));
const { default: __VLS_207 } = __VLS_205.slots;
{
    const { default: __VLS_208 } = __VLS_205.slots;
    const [scope] = __VLS_vSlot(__VLS_208);
    let __VLS_209;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_210 = __VLS_asFunctionalComponent1(__VLS_209, new __VLS_209({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_211 = __VLS_210({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_210));
    let __VLS_214;
    const __VLS_215 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDeleteProductCateRelation(scope.$index);
                // @ts-ignore
                [handleDeleteProductCateRelation,];
            } });
    const { default: __VLS_216 } = __VLS_212.slots;
    // @ts-ignore
    [];
    var __VLS_212;
    var __VLS_213;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_205;
// @ts-ignore
[];
var __VLS_190;
// @ts-ignore
[];
var __VLS_171;
let __VLS_217;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_218 = __VLS_asFunctionalComponent1(__VLS_217, new __VLS_217({}));
const __VLS_219 = __VLS_218({}, ...__VLS_functionalComponentArgsRest(__VLS_218));
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.coupon.useType === 2) }, null, null);
const { default: __VLS_222 } = __VLS_220.slots;
let __VLS_223;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_224 = __VLS_asFunctionalComponent1(__VLS_223, new __VLS_223({
    modelValue: (__VLS_ctx.selectProduct),
    filterable: true,
    remote: true,
    reserveKeyword: true,
    placeholder: "商品名称/商品货号",
    remoteMethod: (__VLS_ctx.searchProductMethod),
    loading: (__VLS_ctx.selectProductLoading),
    ...{ style: {} },
}));
const __VLS_225 = __VLS_224({
    modelValue: (__VLS_ctx.selectProduct),
    filterable: true,
    remote: true,
    reserveKeyword: true,
    placeholder: "商品名称/商品货号",
    remoteMethod: (__VLS_ctx.searchProductMethod),
    loading: (__VLS_ctx.selectProductLoading),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_224));
const { default: __VLS_228 } = __VLS_226.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.selectProductOptions))) {
    let __VLS_229;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_230 = __VLS_asFunctionalComponent1(__VLS_229, new __VLS_229({
        key: (item.productId),
        label: (item.productName),
        value: (item.productId),
    }));
    const __VLS_231 = __VLS_230({
        key: (item.productId),
        label: (item.productName),
        value: (item.productId),
    }, ...__VLS_functionalComponentArgsRest(__VLS_230));
    const { default: __VLS_234 } = __VLS_232.slots;
    {
        const { default: __VLS_235 } = __VLS_232.slots;
        __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
            ...{ style: {} },
        });
        (item.productName);
        __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
            ...{ style: {} },
        });
        (item.productSn);
        // @ts-ignore
        [coupon, selectProduct, searchProductMethod, selectProductLoading, selectProductOptions,];
    }
    // @ts-ignore
    [];
    var __VLS_232;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_226;
let __VLS_236;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_237 = __VLS_asFunctionalComponent1(__VLS_236, new __VLS_236({
    ...{ 'onClick': {} },
}));
const __VLS_238 = __VLS_237({
    ...{ 'onClick': {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_237));
let __VLS_241;
const __VLS_242 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleAddProductRelation();
            // @ts-ignore
            [handleAddProductRelation,];
        } });
const { default: __VLS_243 } = __VLS_239.slots;
// @ts-ignore
[];
var __VLS_239;
var __VLS_240;
let __VLS_244;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_245 = __VLS_asFunctionalComponent1(__VLS_244, new __VLS_244({
    ref: "productRelationTable",
    data: (__VLS_ctx.coupon.productRelationList),
    ...{ style: {} },
    border: true,
}));
const __VLS_246 = __VLS_245({
    ref: "productRelationTable",
    data: (__VLS_ctx.coupon.productRelationList),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_245));
var __VLS_249 = {};
const { default: __VLS_251 } = __VLS_247.slots;
let __VLS_252;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_253 = __VLS_asFunctionalComponent1(__VLS_252, new __VLS_252({
    label: "商品名称",
    align: "center",
}));
const __VLS_254 = __VLS_253({
    label: "商品名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_253));
const { default: __VLS_257 } = __VLS_255.slots;
{
    const { default: __VLS_258 } = __VLS_255.slots;
    const [scope] = __VLS_vSlot(__VLS_258);
    (scope.row.productName);
    // @ts-ignore
    [coupon,];
}
// @ts-ignore
[];
var __VLS_255;
let __VLS_259;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_260 = __VLS_asFunctionalComponent1(__VLS_259, new __VLS_259({
    label: "货号",
    align: "center",
    width: "120",
}));
const __VLS_261 = __VLS_260({
    label: "货号",
    align: "center",
    width: "120",
}, ...__VLS_functionalComponentArgsRest(__VLS_260));
const { default: __VLS_264 } = __VLS_262.slots;
{
    const { default: __VLS_265 } = __VLS_262.slots;
    const [scope] = __VLS_vSlot(__VLS_265);
    (scope.row.productSn);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_262;
let __VLS_266;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_267 = __VLS_asFunctionalComponent1(__VLS_266, new __VLS_266({
    label: "操作",
    align: "center",
    width: "100",
}));
const __VLS_268 = __VLS_267({
    label: "操作",
    align: "center",
    width: "100",
}, ...__VLS_functionalComponentArgsRest(__VLS_267));
const { default: __VLS_271 } = __VLS_269.slots;
{
    const { default: __VLS_272 } = __VLS_269.slots;
    const [scope] = __VLS_vSlot(__VLS_272);
    let __VLS_273;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_274 = __VLS_asFunctionalComponent1(__VLS_273, new __VLS_273({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_275 = __VLS_274({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_274));
    let __VLS_278;
    const __VLS_279 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDeleteProductRelation(scope.$index);
                // @ts-ignore
                [handleDeleteProductRelation,];
            } });
    const { default: __VLS_280 } = __VLS_276.slots;
    // @ts-ignore
    [];
    var __VLS_276;
    var __VLS_277;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_269;
// @ts-ignore
[];
var __VLS_247;
// @ts-ignore
[];
var __VLS_220;
let __VLS_281;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_282 = __VLS_asFunctionalComponent1(__VLS_281, new __VLS_281({
    label: "备注：",
}));
const __VLS_283 = __VLS_282({
    label: "备注：",
}, ...__VLS_functionalComponentArgsRest(__VLS_282));
const { default: __VLS_286 } = __VLS_284.slots;
let __VLS_287;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_288 = __VLS_asFunctionalComponent1(__VLS_287, new __VLS_287({
    ...{ class: "input-width" },
    type: "textarea",
    rows: (5),
    placeholder: "请输入内容",
    modelValue: (__VLS_ctx.coupon.note),
}));
const __VLS_289 = __VLS_288({
    ...{ class: "input-width" },
    type: "textarea",
    rows: (5),
    placeholder: "请输入内容",
    modelValue: (__VLS_ctx.coupon.note),
}, ...__VLS_functionalComponentArgsRest(__VLS_288));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[coupon,];
var __VLS_284;
let __VLS_292;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_293 = __VLS_asFunctionalComponent1(__VLS_292, new __VLS_292({}));
const __VLS_294 = __VLS_293({}, ...__VLS_functionalComponentArgsRest(__VLS_293));
const { default: __VLS_297 } = __VLS_295.slots;
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
    { onClick: (...[$event]) => {
            __VLS_ctx.onSubmit();
            // @ts-ignore
            [onSubmit,];
        } });
const { default: __VLS_305 } = __VLS_301.slots;
// @ts-ignore
[];
var __VLS_301;
var __VLS_302;
if (!__VLS_ctx.isEdit) {
    let __VLS_306;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_307 = __VLS_asFunctionalComponent1(__VLS_306, new __VLS_306({
        ...{ 'onClick': {} },
    }));
    const __VLS_308 = __VLS_307({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_307));
    let __VLS_311;
    const __VLS_312 = ({ click: {} },
        { onClick: (...[$event]) => {
                if (!(!__VLS_ctx.isEdit))
                    return;
                __VLS_ctx.resetForm();
                // @ts-ignore
                [isEdit, resetForm,];
            } });
    const { default: __VLS_313 } = __VLS_309.slots;
    // @ts-ignore
    [];
    var __VLS_309;
    var __VLS_310;
}
// @ts-ignore
[];
var __VLS_295;
// @ts-ignore
[];
var __VLS_10;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
var __VLS_13 = __VLS_12, __VLS_193 = __VLS_192, __VLS_250 = __VLS_249;
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
