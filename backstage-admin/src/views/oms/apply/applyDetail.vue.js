/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, computed, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getReturnApplyByIdAPI, returnApplyUpdateStatusAPI } from '@/apis/returnApply';
import { getCompanyAddressListAPI } from '@/apis/companyAddress';
import { formatDateTime } from '@/utils/datetime';
// 默认状态修改参数
const defaultUpdateStatusParam = {
    id: 0,
    companyAddressId: 0,
    handleMan: 'admin',
    handleNote: '',
    receiveMan: 'admin',
    receiveNote: '',
    returnAmount: 0,
    status: 0
};
// 路由相关
const route = useRoute();
const router = useRouter();
// 当前退货申请ID
const id = ref();
// 当前退货申请
const orderReturnApply = ref({});
// 凭证图片
const proofPics = ref([]);
// 退货商品列表
const productList = ref();
// 公司收货地址列表
const companyAddressList = ref([]);
// 修改退货申请状态参数
const updateStatusParam = ref(Object.assign({}, defaultUpdateStatusParam));
// 获取详情
const getDetail = async () => {
    const res = await getReturnApplyByIdAPI(id.value);
    orderReturnApply.value = res.data;
    productList.value = [];
    productList.value.push(orderReturnApply.value);
    if (orderReturnApply.value.proofPics) {
        proofPics.value = orderReturnApply.value.proofPics.split(",");
    }
    // 退货中和完成
    if (orderReturnApply.value.status === 1 || orderReturnApply.value.status === 2) {
        updateStatusParam.value.returnAmount = orderReturnApply.value.returnAmount;
        updateStatusParam.value.companyAddressId = orderReturnApply.value.companyAddressId;
    }
};
// 获取公司地址列表
const getCompanyAddressList = async () => {
    const res = await getCompanyAddressListAPI();
    companyAddressList.value = res.data;
    // 获取默认收货地址
    const defaultAddress = companyAddressList.value.find(item => item.receiveStatus === 1);
    if (defaultAddress) {
        updateStatusParam.value.companyAddressId = defaultAddress.id;
    }
};
// 组件挂载
onMounted(() => {
    id.value = route.query.id;
    getDetail();
    getCompanyAddressList();
});
// 计算属性
const totalAmount = computed(() => {
    if (orderReturnApply.value != null) {
        return orderReturnApply.value.productRealPrice * orderReturnApply.value.productCount;
    }
    else {
        return 0;
    }
});
// 当前收货地址
const currentAddress = computed(() => {
    const idValue = updateStatusParam.value.companyAddressId;
    if (!companyAddressList.value)
        return undefined;
    return companyAddressList.value.find(item => item.id === idValue);
});
// 格式化状态
const formatStatus = (status) => {
    if (status === 0) {
        return "待处理";
    }
    else if (status === 1) {
        return "退货中";
    }
    else if (status === 2) {
        return "已完成";
    }
    else {
        return "已拒绝";
    }
};
// 格式化地区
const formatRegion = (address) => {
    if (!address)
        return '';
    let str = address.province;
    if (address.city) {
        str += "  " + address.city;
    }
    str += "  " + address.region;
    return str;
};
// 查看订单详情
const handleViewOrder = () => {
    router.push({ path: '/oms/orderDetail', query: { id: orderReturnApply.value.orderId } });
};
// 更新状态
const handleUpdateStatus = async (status) => {
    updateStatusParam.value.status = status;
    await ElMessageBox.confirm('是否要进行此操作?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    await returnApplyUpdateStatusAPI(id.value, updateStatusParam.value);
    ElMessage({
        type: 'success',
        message: '操作成功!',
        duration: 1000
    });
    router.back();
};
const __VLS_ctx = {
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "detail-container" },
});
/** @type {__VLS_StyleScopedClasses['detail-container']} */ ;
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    shadow: "never",
}));
const __VLS_2 = __VLS_1({
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
const { default: __VLS_5 } = __VLS_3.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-title-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-title-medium']} */ ;
let __VLS_6;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({
    border: true,
    ...{ class: "standard-margin" },
    ref: "productTable",
    data: (__VLS_ctx.productList),
}));
const __VLS_8 = __VLS_7({
    border: true,
    ...{ class: "standard-margin" },
    ref: "productTable",
    data: (__VLS_ctx.productList),
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
var __VLS_11 = {};
/** @type {__VLS_StyleScopedClasses['standard-margin']} */ ;
const { default: __VLS_13 } = __VLS_9.slots;
let __VLS_14;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_15 = __VLS_asFunctionalComponent1(__VLS_14, new __VLS_14({
    label: "商品图片",
    width: "160",
    align: "center",
}));
const __VLS_16 = __VLS_15({
    label: "商品图片",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
const { default: __VLS_19 } = __VLS_17.slots;
{
    const { default: __VLS_20 } = __VLS_17.slots;
    const [scope] = __VLS_vSlot(__VLS_20);
    __VLS_asFunctionalElement1(__VLS_intrinsics.img)({
        ...{ style: {} },
        src: (scope.row.productPic),
    });
    // @ts-ignore
    [productList,];
}
// @ts-ignore
[];
var __VLS_17;
let __VLS_21;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_22 = __VLS_asFunctionalComponent1(__VLS_21, new __VLS_21({
    label: "商品名称",
    align: "center",
}));
const __VLS_23 = __VLS_22({
    label: "商品名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_22));
const { default: __VLS_26 } = __VLS_24.slots;
{
    const { default: __VLS_27 } = __VLS_24.slots;
    const [scope] = __VLS_vSlot(__VLS_27);
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "font-small" },
    });
    /** @type {__VLS_StyleScopedClasses['font-small']} */ ;
    (scope.row.productName);
    __VLS_asFunctionalElement1(__VLS_intrinsics.br)({});
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "font-small" },
    });
    /** @type {__VLS_StyleScopedClasses['font-small']} */ ;
    (scope.row.productBrand);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_24;
let __VLS_28;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_29 = __VLS_asFunctionalComponent1(__VLS_28, new __VLS_28({
    label: "价格/货号",
    width: "180",
    align: "center",
}));
const __VLS_30 = __VLS_29({
    label: "价格/货号",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_29));
const { default: __VLS_33 } = __VLS_31.slots;
{
    const { default: __VLS_34 } = __VLS_31.slots;
    const [scope] = __VLS_vSlot(__VLS_34);
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "font-small" },
    });
    /** @type {__VLS_StyleScopedClasses['font-small']} */ ;
    (scope.row.productRealPrice);
    __VLS_asFunctionalElement1(__VLS_intrinsics.br)({});
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "font-small" },
    });
    /** @type {__VLS_StyleScopedClasses['font-small']} */ ;
    (scope.row.productId);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_31;
let __VLS_35;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_36 = __VLS_asFunctionalComponent1(__VLS_35, new __VLS_35({
    label: "属性",
    width: "180",
    align: "center",
}));
const __VLS_37 = __VLS_36({
    label: "属性",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_36));
const { default: __VLS_40 } = __VLS_38.slots;
{
    const { default: __VLS_41 } = __VLS_38.slots;
    const [scope] = __VLS_vSlot(__VLS_41);
    (scope.row.productAttr);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_38;
let __VLS_42;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_43 = __VLS_asFunctionalComponent1(__VLS_42, new __VLS_42({
    label: "数量",
    width: "100",
    align: "center",
}));
const __VLS_44 = __VLS_43({
    label: "数量",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_43));
const { default: __VLS_47 } = __VLS_45.slots;
{
    const { default: __VLS_48 } = __VLS_45.slots;
    const [scope] = __VLS_vSlot(__VLS_48);
    (scope.row.productCount);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_45;
let __VLS_49;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_50 = __VLS_asFunctionalComponent1(__VLS_49, new __VLS_49({
    label: "小计",
    width: "100",
    align: "center",
}));
const __VLS_51 = __VLS_50({
    label: "小计",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_50));
const { default: __VLS_54 } = __VLS_52.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.template, __VLS_intrinsics.template)({});
(__VLS_ctx.totalAmount);
// @ts-ignore
[totalAmount,];
var __VLS_52;
// @ts-ignore
[];
var __VLS_9;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-title-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-title-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-title-medium color-danger" },
});
/** @type {__VLS_StyleScopedClasses['font-title-medium']} */ ;
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
(__VLS_ctx.totalAmount);
// @ts-ignore
[totalAmount,];
var __VLS_3;
let __VLS_55;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_56 = __VLS_asFunctionalComponent1(__VLS_55, new __VLS_55({
    shadow: "never",
    ...{ class: "standard-margin" },
}));
const __VLS_57 = __VLS_56({
    shadow: "never",
    ...{ class: "standard-margin" },
}, ...__VLS_functionalComponentArgsRest(__VLS_56));
/** @type {__VLS_StyleScopedClasses['standard-margin']} */ ;
const { default: __VLS_60 } = __VLS_58.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-title-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-title-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "form-container-border" },
});
/** @type {__VLS_StyleScopedClasses['form-container-border']} */ ;
let __VLS_61;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_62 = __VLS_asFunctionalComponent1(__VLS_61, new __VLS_61({}));
const __VLS_63 = __VLS_62({}, ...__VLS_functionalComponentArgsRest(__VLS_62));
const { default: __VLS_66 } = __VLS_64.slots;
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    span: (6),
    ...{ class: "form-border form-left-bg font-small" },
}));
const __VLS_69 = __VLS_68({
    span: (6),
    ...{ class: "form-border form-left-bg font-small" },
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_72 } = __VLS_70.slots;
// @ts-ignore
[];
var __VLS_70;
let __VLS_73;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_74 = __VLS_asFunctionalComponent1(__VLS_73, new __VLS_73({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_75 = __VLS_74({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_74));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_78 } = __VLS_76.slots;
(__VLS_ctx.orderReturnApply.id);
// @ts-ignore
[orderReturnApply,];
var __VLS_76;
// @ts-ignore
[];
var __VLS_64;
let __VLS_79;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_80 = __VLS_asFunctionalComponent1(__VLS_79, new __VLS_79({}));
const __VLS_81 = __VLS_80({}, ...__VLS_functionalComponentArgsRest(__VLS_80));
const { default: __VLS_84 } = __VLS_82.slots;
let __VLS_85;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_86 = __VLS_asFunctionalComponent1(__VLS_85, new __VLS_85({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_87 = __VLS_86({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_86));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_90 } = __VLS_88.slots;
// @ts-ignore
[];
var __VLS_88;
let __VLS_91;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_92 = __VLS_asFunctionalComponent1(__VLS_91, new __VLS_91({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_93 = __VLS_92({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_92));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_96 } = __VLS_94.slots;
(__VLS_ctx.formatStatus(__VLS_ctx.orderReturnApply.status));
// @ts-ignore
[orderReturnApply, formatStatus,];
var __VLS_94;
// @ts-ignore
[];
var __VLS_82;
let __VLS_97;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_98 = __VLS_asFunctionalComponent1(__VLS_97, new __VLS_97({}));
const __VLS_99 = __VLS_98({}, ...__VLS_functionalComponentArgsRest(__VLS_98));
const { default: __VLS_102 } = __VLS_100.slots;
let __VLS_103;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_104 = __VLS_asFunctionalComponent1(__VLS_103, new __VLS_103({
    span: (6),
    ...{ class: "form-border form-left-bg font-small" },
    ...{ style: {} },
}));
const __VLS_105 = __VLS_104({
    span: (6),
    ...{ class: "form-border form-left-bg font-small" },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_104));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_108 } = __VLS_106.slots;
// @ts-ignore
[];
var __VLS_106;
let __VLS_109;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_110 = __VLS_asFunctionalComponent1(__VLS_109, new __VLS_109({
    ...{ class: "form-border font-small" },
    span: (18),
    ...{ style: {} },
}));
const __VLS_111 = __VLS_110({
    ...{ class: "form-border font-small" },
    span: (18),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_110));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_114 } = __VLS_112.slots;
(__VLS_ctx.orderReturnApply.orderSn);
let __VLS_115;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_116 = __VLS_asFunctionalComponent1(__VLS_115, new __VLS_115({
    ...{ 'onClick': {} },
    type: "text",
    size: "small",
}));
const __VLS_117 = __VLS_116({
    ...{ 'onClick': {} },
    type: "text",
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_116));
let __VLS_120;
const __VLS_121 = ({ click: {} },
    { onClick: (__VLS_ctx.handleViewOrder) });
const { default: __VLS_122 } = __VLS_118.slots;
// @ts-ignore
[orderReturnApply, handleViewOrder,];
var __VLS_118;
var __VLS_119;
// @ts-ignore
[];
var __VLS_112;
// @ts-ignore
[];
var __VLS_100;
let __VLS_123;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_124 = __VLS_asFunctionalComponent1(__VLS_123, new __VLS_123({}));
const __VLS_125 = __VLS_124({}, ...__VLS_functionalComponentArgsRest(__VLS_124));
const { default: __VLS_128 } = __VLS_126.slots;
let __VLS_129;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_130 = __VLS_asFunctionalComponent1(__VLS_129, new __VLS_129({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_131 = __VLS_130({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_130));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_134 } = __VLS_132.slots;
// @ts-ignore
[];
var __VLS_132;
let __VLS_135;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_136 = __VLS_asFunctionalComponent1(__VLS_135, new __VLS_135({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_137 = __VLS_136({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_136));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_140 } = __VLS_138.slots;
(__VLS_ctx.formatDateTime(__VLS_ctx.orderReturnApply.createTime));
// @ts-ignore
[orderReturnApply, formatDateTime,];
var __VLS_138;
// @ts-ignore
[];
var __VLS_126;
let __VLS_141;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_142 = __VLS_asFunctionalComponent1(__VLS_141, new __VLS_141({}));
const __VLS_143 = __VLS_142({}, ...__VLS_functionalComponentArgsRest(__VLS_142));
const { default: __VLS_146 } = __VLS_144.slots;
let __VLS_147;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_148 = __VLS_asFunctionalComponent1(__VLS_147, new __VLS_147({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_149 = __VLS_148({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_148));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_152 } = __VLS_150.slots;
// @ts-ignore
[];
var __VLS_150;
let __VLS_153;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_154 = __VLS_asFunctionalComponent1(__VLS_153, new __VLS_153({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_155 = __VLS_154({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_154));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_158 } = __VLS_156.slots;
(__VLS_ctx.orderReturnApply.memberUsername);
// @ts-ignore
[orderReturnApply,];
var __VLS_156;
// @ts-ignore
[];
var __VLS_144;
let __VLS_159;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_160 = __VLS_asFunctionalComponent1(__VLS_159, new __VLS_159({}));
const __VLS_161 = __VLS_160({}, ...__VLS_functionalComponentArgsRest(__VLS_160));
const { default: __VLS_164 } = __VLS_162.slots;
let __VLS_165;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_166 = __VLS_asFunctionalComponent1(__VLS_165, new __VLS_165({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_167 = __VLS_166({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_166));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_170 } = __VLS_168.slots;
// @ts-ignore
[];
var __VLS_168;
let __VLS_171;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_172 = __VLS_asFunctionalComponent1(__VLS_171, new __VLS_171({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_173 = __VLS_172({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_172));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_176 } = __VLS_174.slots;
(__VLS_ctx.orderReturnApply.returnName);
// @ts-ignore
[orderReturnApply,];
var __VLS_174;
// @ts-ignore
[];
var __VLS_162;
let __VLS_177;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_178 = __VLS_asFunctionalComponent1(__VLS_177, new __VLS_177({}));
const __VLS_179 = __VLS_178({}, ...__VLS_functionalComponentArgsRest(__VLS_178));
const { default: __VLS_182 } = __VLS_180.slots;
let __VLS_183;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_184 = __VLS_asFunctionalComponent1(__VLS_183, new __VLS_183({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_185 = __VLS_184({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_184));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_188 } = __VLS_186.slots;
// @ts-ignore
[];
var __VLS_186;
let __VLS_189;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_190 = __VLS_asFunctionalComponent1(__VLS_189, new __VLS_189({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_191 = __VLS_190({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_190));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_194 } = __VLS_192.slots;
(__VLS_ctx.orderReturnApply.returnPhone);
// @ts-ignore
[orderReturnApply,];
var __VLS_192;
// @ts-ignore
[];
var __VLS_180;
let __VLS_195;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_196 = __VLS_asFunctionalComponent1(__VLS_195, new __VLS_195({}));
const __VLS_197 = __VLS_196({}, ...__VLS_functionalComponentArgsRest(__VLS_196));
const { default: __VLS_200 } = __VLS_198.slots;
let __VLS_201;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_202 = __VLS_asFunctionalComponent1(__VLS_201, new __VLS_201({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_203 = __VLS_202({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_202));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_206 } = __VLS_204.slots;
// @ts-ignore
[];
var __VLS_204;
let __VLS_207;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_208 = __VLS_asFunctionalComponent1(__VLS_207, new __VLS_207({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_209 = __VLS_208({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_208));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_212 } = __VLS_210.slots;
(__VLS_ctx.orderReturnApply.reason);
// @ts-ignore
[orderReturnApply,];
var __VLS_210;
// @ts-ignore
[];
var __VLS_198;
let __VLS_213;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_214 = __VLS_asFunctionalComponent1(__VLS_213, new __VLS_213({}));
const __VLS_215 = __VLS_214({}, ...__VLS_functionalComponentArgsRest(__VLS_214));
const { default: __VLS_218 } = __VLS_216.slots;
let __VLS_219;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_220 = __VLS_asFunctionalComponent1(__VLS_219, new __VLS_219({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_221 = __VLS_220({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_220));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_224 } = __VLS_222.slots;
// @ts-ignore
[];
var __VLS_222;
let __VLS_225;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_226 = __VLS_asFunctionalComponent1(__VLS_225, new __VLS_225({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_227 = __VLS_226({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_226));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_230 } = __VLS_228.slots;
(__VLS_ctx.orderReturnApply.description);
// @ts-ignore
[orderReturnApply,];
var __VLS_228;
// @ts-ignore
[];
var __VLS_216;
let __VLS_231;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_232 = __VLS_asFunctionalComponent1(__VLS_231, new __VLS_231({}));
const __VLS_233 = __VLS_232({}, ...__VLS_functionalComponentArgsRest(__VLS_232));
const { default: __VLS_236 } = __VLS_234.slots;
let __VLS_237;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_238 = __VLS_asFunctionalComponent1(__VLS_237, new __VLS_237({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}));
const __VLS_239 = __VLS_238({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_238));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_242 } = __VLS_240.slots;
// @ts-ignore
[];
var __VLS_240;
let __VLS_243;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_244 = __VLS_asFunctionalComponent1(__VLS_243, new __VLS_243({
    ...{ class: "form-border font-small" },
    span: (18),
    ...{ style: {} },
}));
const __VLS_245 = __VLS_244({
    ...{ class: "form-border font-small" },
    span: (18),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_244));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_248 } = __VLS_246.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.proofPics))) {
    __VLS_asFunctionalElement1(__VLS_intrinsics.img)({
        ...{ style: {} },
        src: (item),
        key: (item),
    });
    // @ts-ignore
    [proofPics,];
}
// @ts-ignore
[];
var __VLS_246;
// @ts-ignore
[];
var __VLS_234;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "form-container-border" },
});
/** @type {__VLS_StyleScopedClasses['form-container-border']} */ ;
let __VLS_249;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_250 = __VLS_asFunctionalComponent1(__VLS_249, new __VLS_249({}));
const __VLS_251 = __VLS_250({}, ...__VLS_functionalComponentArgsRest(__VLS_250));
const { default: __VLS_254 } = __VLS_252.slots;
let __VLS_255;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_256 = __VLS_asFunctionalComponent1(__VLS_255, new __VLS_255({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_257 = __VLS_256({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_256));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_260 } = __VLS_258.slots;
// @ts-ignore
[];
var __VLS_258;
let __VLS_261;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_262 = __VLS_asFunctionalComponent1(__VLS_261, new __VLS_261({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_263 = __VLS_262({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_262));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_266 } = __VLS_264.slots;
(__VLS_ctx.totalAmount);
// @ts-ignore
[totalAmount,];
var __VLS_264;
// @ts-ignore
[];
var __VLS_252;
let __VLS_267;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_268 = __VLS_asFunctionalComponent1(__VLS_267, new __VLS_267({}));
const __VLS_269 = __VLS_268({}, ...__VLS_functionalComponentArgsRest(__VLS_268));
const { default: __VLS_272 } = __VLS_270.slots;
let __VLS_273;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_274 = __VLS_asFunctionalComponent1(__VLS_273, new __VLS_273({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}));
const __VLS_275 = __VLS_274({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_274));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_278 } = __VLS_276.slots;
// @ts-ignore
[];
var __VLS_276;
let __VLS_279;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_280 = __VLS_asFunctionalComponent1(__VLS_279, new __VLS_279({
    ...{ class: "form-border font-small" },
    ...{ style: {} },
    span: (18),
}));
const __VLS_281 = __VLS_280({
    ...{ class: "form-border font-small" },
    ...{ style: {} },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_280));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_284 } = __VLS_282.slots;
let __VLS_285;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_286 = __VLS_asFunctionalComponent1(__VLS_285, new __VLS_285({
    size: "small",
    modelValue: (__VLS_ctx.updateStatusParam.returnAmount),
    disabled: (__VLS_ctx.orderReturnApply.status !== 0),
    ...{ style: {} },
}));
const __VLS_287 = __VLS_286({
    size: "small",
    modelValue: (__VLS_ctx.updateStatusParam.returnAmount),
    disabled: (__VLS_ctx.orderReturnApply.status !== 0),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_286));
// @ts-ignore
[orderReturnApply, updateStatusParam,];
var __VLS_282;
// @ts-ignore
[];
var __VLS_270;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.orderReturnApply.status !== 3) }, null, null);
let __VLS_290;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_291 = __VLS_asFunctionalComponent1(__VLS_290, new __VLS_290({}));
const __VLS_292 = __VLS_291({}, ...__VLS_functionalComponentArgsRest(__VLS_291));
const { default: __VLS_295 } = __VLS_293.slots;
let __VLS_296;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_297 = __VLS_asFunctionalComponent1(__VLS_296, new __VLS_296({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}));
const __VLS_298 = __VLS_297({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_297));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_301 } = __VLS_299.slots;
// @ts-ignore
[orderReturnApply,];
var __VLS_299;
let __VLS_302;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_303 = __VLS_asFunctionalComponent1(__VLS_302, new __VLS_302({
    ...{ class: "form-border font-small" },
    ...{ style: {} },
    span: (18),
}));
const __VLS_304 = __VLS_303({
    ...{ class: "form-border font-small" },
    ...{ style: {} },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_303));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_307 } = __VLS_305.slots;
let __VLS_308;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_309 = __VLS_asFunctionalComponent1(__VLS_308, new __VLS_308({
    size: "small",
    ...{ style: {} },
    disabled: (__VLS_ctx.orderReturnApply.status !== 0),
    modelValue: (__VLS_ctx.updateStatusParam.companyAddressId),
}));
const __VLS_310 = __VLS_309({
    size: "small",
    ...{ style: {} },
    disabled: (__VLS_ctx.orderReturnApply.status !== 0),
    modelValue: (__VLS_ctx.updateStatusParam.companyAddressId),
}, ...__VLS_functionalComponentArgsRest(__VLS_309));
const { default: __VLS_313 } = __VLS_311.slots;
for (const [address] of __VLS_vFor((__VLS_ctx.companyAddressList))) {
    let __VLS_314;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_315 = __VLS_asFunctionalComponent1(__VLS_314, new __VLS_314({
        key: (address.id),
        value: (address.id),
        label: (address.addressName),
    }));
    const __VLS_316 = __VLS_315({
        key: (address.id),
        value: (address.id),
        label: (address.addressName),
    }, ...__VLS_functionalComponentArgsRest(__VLS_315));
    // @ts-ignore
    [orderReturnApply, updateStatusParam, companyAddressList,];
}
// @ts-ignore
[];
var __VLS_311;
// @ts-ignore
[];
var __VLS_305;
// @ts-ignore
[];
var __VLS_293;
let __VLS_319;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_320 = __VLS_asFunctionalComponent1(__VLS_319, new __VLS_319({}));
const __VLS_321 = __VLS_320({}, ...__VLS_functionalComponentArgsRest(__VLS_320));
const { default: __VLS_324 } = __VLS_322.slots;
let __VLS_325;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_326 = __VLS_asFunctionalComponent1(__VLS_325, new __VLS_325({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_327 = __VLS_326({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_326));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_330 } = __VLS_328.slots;
// @ts-ignore
[];
var __VLS_328;
let __VLS_331;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_332 = __VLS_asFunctionalComponent1(__VLS_331, new __VLS_331({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_333 = __VLS_332({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_332));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_336 } = __VLS_334.slots;
(__VLS_ctx.currentAddress?.name);
// @ts-ignore
[currentAddress,];
var __VLS_334;
// @ts-ignore
[];
var __VLS_322;
let __VLS_337;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_338 = __VLS_asFunctionalComponent1(__VLS_337, new __VLS_337({}));
const __VLS_339 = __VLS_338({}, ...__VLS_functionalComponentArgsRest(__VLS_338));
const { default: __VLS_342 } = __VLS_340.slots;
let __VLS_343;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_344 = __VLS_asFunctionalComponent1(__VLS_343, new __VLS_343({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_345 = __VLS_344({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_344));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_348 } = __VLS_346.slots;
// @ts-ignore
[];
var __VLS_346;
let __VLS_349;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_350 = __VLS_asFunctionalComponent1(__VLS_349, new __VLS_349({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_351 = __VLS_350({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_350));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_354 } = __VLS_352.slots;
(__VLS_ctx.formatRegion(__VLS_ctx.currentAddress));
// @ts-ignore
[currentAddress, formatRegion,];
var __VLS_352;
// @ts-ignore
[];
var __VLS_340;
let __VLS_355;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_356 = __VLS_asFunctionalComponent1(__VLS_355, new __VLS_355({}));
const __VLS_357 = __VLS_356({}, ...__VLS_functionalComponentArgsRest(__VLS_356));
const { default: __VLS_360 } = __VLS_358.slots;
let __VLS_361;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_362 = __VLS_asFunctionalComponent1(__VLS_361, new __VLS_361({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_363 = __VLS_362({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_362));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_366 } = __VLS_364.slots;
// @ts-ignore
[];
var __VLS_364;
let __VLS_367;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_368 = __VLS_asFunctionalComponent1(__VLS_367, new __VLS_367({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_369 = __VLS_368({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_368));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_372 } = __VLS_370.slots;
(__VLS_ctx.currentAddress?.detailAddress);
// @ts-ignore
[currentAddress,];
var __VLS_370;
// @ts-ignore
[];
var __VLS_358;
let __VLS_373;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_374 = __VLS_asFunctionalComponent1(__VLS_373, new __VLS_373({}));
const __VLS_375 = __VLS_374({}, ...__VLS_functionalComponentArgsRest(__VLS_374));
const { default: __VLS_378 } = __VLS_376.slots;
let __VLS_379;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_380 = __VLS_asFunctionalComponent1(__VLS_379, new __VLS_379({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_381 = __VLS_380({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_380));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_384 } = __VLS_382.slots;
// @ts-ignore
[];
var __VLS_382;
let __VLS_385;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_386 = __VLS_asFunctionalComponent1(__VLS_385, new __VLS_385({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_387 = __VLS_386({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_386));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_390 } = __VLS_388.slots;
(__VLS_ctx.currentAddress?.phone);
// @ts-ignore
[currentAddress,];
var __VLS_388;
// @ts-ignore
[];
var __VLS_376;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "form-container-border" },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.orderReturnApply.status !== 0) }, null, null);
/** @type {__VLS_StyleScopedClasses['form-container-border']} */ ;
let __VLS_391;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_392 = __VLS_asFunctionalComponent1(__VLS_391, new __VLS_391({}));
const __VLS_393 = __VLS_392({}, ...__VLS_functionalComponentArgsRest(__VLS_392));
const { default: __VLS_396 } = __VLS_394.slots;
let __VLS_397;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_398 = __VLS_asFunctionalComponent1(__VLS_397, new __VLS_397({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_399 = __VLS_398({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_398));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_402 } = __VLS_400.slots;
// @ts-ignore
[orderReturnApply,];
var __VLS_400;
let __VLS_403;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_404 = __VLS_asFunctionalComponent1(__VLS_403, new __VLS_403({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_405 = __VLS_404({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_404));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_408 } = __VLS_406.slots;
(__VLS_ctx.orderReturnApply.handleMan);
// @ts-ignore
[orderReturnApply,];
var __VLS_406;
// @ts-ignore
[];
var __VLS_394;
let __VLS_409;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_410 = __VLS_asFunctionalComponent1(__VLS_409, new __VLS_409({}));
const __VLS_411 = __VLS_410({}, ...__VLS_functionalComponentArgsRest(__VLS_410));
const { default: __VLS_414 } = __VLS_412.slots;
let __VLS_415;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_416 = __VLS_asFunctionalComponent1(__VLS_415, new __VLS_415({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_417 = __VLS_416({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_416));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_420 } = __VLS_418.slots;
// @ts-ignore
[];
var __VLS_418;
let __VLS_421;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_422 = __VLS_asFunctionalComponent1(__VLS_421, new __VLS_421({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_423 = __VLS_422({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_422));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_426 } = __VLS_424.slots;
(__VLS_ctx.formatDateTime(__VLS_ctx.orderReturnApply.handleTime));
// @ts-ignore
[orderReturnApply, formatDateTime,];
var __VLS_424;
// @ts-ignore
[];
var __VLS_412;
let __VLS_427;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_428 = __VLS_asFunctionalComponent1(__VLS_427, new __VLS_427({}));
const __VLS_429 = __VLS_428({}, ...__VLS_functionalComponentArgsRest(__VLS_428));
const { default: __VLS_432 } = __VLS_430.slots;
let __VLS_433;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_434 = __VLS_asFunctionalComponent1(__VLS_433, new __VLS_433({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_435 = __VLS_434({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_434));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_438 } = __VLS_436.slots;
// @ts-ignore
[];
var __VLS_436;
let __VLS_439;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_440 = __VLS_asFunctionalComponent1(__VLS_439, new __VLS_439({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_441 = __VLS_440({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_440));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_444 } = __VLS_442.slots;
(__VLS_ctx.orderReturnApply.handleNote);
// @ts-ignore
[orderReturnApply,];
var __VLS_442;
// @ts-ignore
[];
var __VLS_430;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "form-container-border" },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.orderReturnApply.status === 2) }, null, null);
/** @type {__VLS_StyleScopedClasses['form-container-border']} */ ;
let __VLS_445;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_446 = __VLS_asFunctionalComponent1(__VLS_445, new __VLS_445({}));
const __VLS_447 = __VLS_446({}, ...__VLS_functionalComponentArgsRest(__VLS_446));
const { default: __VLS_450 } = __VLS_448.slots;
let __VLS_451;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_452 = __VLS_asFunctionalComponent1(__VLS_451, new __VLS_451({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_453 = __VLS_452({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_452));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_456 } = __VLS_454.slots;
// @ts-ignore
[orderReturnApply,];
var __VLS_454;
let __VLS_457;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_458 = __VLS_asFunctionalComponent1(__VLS_457, new __VLS_457({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_459 = __VLS_458({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_458));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_462 } = __VLS_460.slots;
(__VLS_ctx.orderReturnApply.receiveMan);
// @ts-ignore
[orderReturnApply,];
var __VLS_460;
// @ts-ignore
[];
var __VLS_448;
let __VLS_463;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_464 = __VLS_asFunctionalComponent1(__VLS_463, new __VLS_463({}));
const __VLS_465 = __VLS_464({}, ...__VLS_functionalComponentArgsRest(__VLS_464));
const { default: __VLS_468 } = __VLS_466.slots;
let __VLS_469;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_470 = __VLS_asFunctionalComponent1(__VLS_469, new __VLS_469({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_471 = __VLS_470({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_470));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_474 } = __VLS_472.slots;
// @ts-ignore
[];
var __VLS_472;
let __VLS_475;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_476 = __VLS_asFunctionalComponent1(__VLS_475, new __VLS_475({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_477 = __VLS_476({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_476));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_480 } = __VLS_478.slots;
(__VLS_ctx.formatDateTime(__VLS_ctx.orderReturnApply.receiveTime));
// @ts-ignore
[orderReturnApply, formatDateTime,];
var __VLS_478;
// @ts-ignore
[];
var __VLS_466;
let __VLS_481;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_482 = __VLS_asFunctionalComponent1(__VLS_481, new __VLS_481({}));
const __VLS_483 = __VLS_482({}, ...__VLS_functionalComponentArgsRest(__VLS_482));
const { default: __VLS_486 } = __VLS_484.slots;
let __VLS_487;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_488 = __VLS_asFunctionalComponent1(__VLS_487, new __VLS_487({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}));
const __VLS_489 = __VLS_488({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_488));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_492 } = __VLS_490.slots;
// @ts-ignore
[];
var __VLS_490;
let __VLS_493;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_494 = __VLS_asFunctionalComponent1(__VLS_493, new __VLS_493({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_495 = __VLS_494({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_494));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_498 } = __VLS_496.slots;
(__VLS_ctx.orderReturnApply.receiveNote);
// @ts-ignore
[orderReturnApply,];
var __VLS_496;
// @ts-ignore
[];
var __VLS_484;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "form-container-border" },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.orderReturnApply.status === 0) }, null, null);
/** @type {__VLS_StyleScopedClasses['form-container-border']} */ ;
let __VLS_499;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_500 = __VLS_asFunctionalComponent1(__VLS_499, new __VLS_499({}));
const __VLS_501 = __VLS_500({}, ...__VLS_functionalComponentArgsRest(__VLS_500));
const { default: __VLS_504 } = __VLS_502.slots;
let __VLS_505;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_506 = __VLS_asFunctionalComponent1(__VLS_505, new __VLS_505({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}));
const __VLS_507 = __VLS_506({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_506));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_510 } = __VLS_508.slots;
// @ts-ignore
[orderReturnApply,];
var __VLS_508;
let __VLS_511;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_512 = __VLS_asFunctionalComponent1(__VLS_511, new __VLS_511({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_513 = __VLS_512({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_512));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_516 } = __VLS_514.slots;
let __VLS_517;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_518 = __VLS_asFunctionalComponent1(__VLS_517, new __VLS_517({
    size: "small",
    modelValue: (__VLS_ctx.updateStatusParam.handleNote),
    ...{ style: {} },
}));
const __VLS_519 = __VLS_518({
    size: "small",
    modelValue: (__VLS_ctx.updateStatusParam.handleNote),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_518));
// @ts-ignore
[updateStatusParam,];
var __VLS_514;
// @ts-ignore
[];
var __VLS_502;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "form-container-border" },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.orderReturnApply.status === 1) }, null, null);
/** @type {__VLS_StyleScopedClasses['form-container-border']} */ ;
let __VLS_522;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_523 = __VLS_asFunctionalComponent1(__VLS_522, new __VLS_522({}));
const __VLS_524 = __VLS_523({}, ...__VLS_functionalComponentArgsRest(__VLS_523));
const { default: __VLS_527 } = __VLS_525.slots;
let __VLS_528;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_529 = __VLS_asFunctionalComponent1(__VLS_528, new __VLS_528({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}));
const __VLS_530 = __VLS_529({
    ...{ class: "form-border form-left-bg font-small" },
    span: (6),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_529));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['form-left-bg']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_533 } = __VLS_531.slots;
// @ts-ignore
[orderReturnApply,];
var __VLS_531;
let __VLS_534;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_535 = __VLS_asFunctionalComponent1(__VLS_534, new __VLS_534({
    ...{ class: "form-border font-small" },
    span: (18),
}));
const __VLS_536 = __VLS_535({
    ...{ class: "form-border font-small" },
    span: (18),
}, ...__VLS_functionalComponentArgsRest(__VLS_535));
/** @type {__VLS_StyleScopedClasses['form-border']} */ ;
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
const { default: __VLS_539 } = __VLS_537.slots;
let __VLS_540;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_541 = __VLS_asFunctionalComponent1(__VLS_540, new __VLS_540({
    size: "small",
    modelValue: (__VLS_ctx.updateStatusParam.receiveNote),
    ...{ style: {} },
}));
const __VLS_542 = __VLS_541({
    size: "small",
    modelValue: (__VLS_ctx.updateStatusParam.receiveNote),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_541));
// @ts-ignore
[updateStatusParam,];
var __VLS_537;
// @ts-ignore
[];
var __VLS_525;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.orderReturnApply.status === 0) }, null, null);
let __VLS_545;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_546 = __VLS_asFunctionalComponent1(__VLS_545, new __VLS_545({
    ...{ 'onClick': {} },
    type: "primary",
    size: "small",
}));
const __VLS_547 = __VLS_546({
    ...{ 'onClick': {} },
    type: "primary",
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_546));
let __VLS_550;
const __VLS_551 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleUpdateStatus(1);
            // @ts-ignore
            [orderReturnApply, handleUpdateStatus,];
        } });
const { default: __VLS_552 } = __VLS_548.slots;
// @ts-ignore
[];
var __VLS_548;
var __VLS_549;
let __VLS_553;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_554 = __VLS_asFunctionalComponent1(__VLS_553, new __VLS_553({
    ...{ 'onClick': {} },
    type: "danger",
    size: "small",
}));
const __VLS_555 = __VLS_554({
    ...{ 'onClick': {} },
    type: "danger",
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_554));
let __VLS_558;
const __VLS_559 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleUpdateStatus(3);
            // @ts-ignore
            [handleUpdateStatus,];
        } });
const { default: __VLS_560 } = __VLS_556.slots;
// @ts-ignore
[];
var __VLS_556;
var __VLS_557;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.orderReturnApply.status === 1) }, null, null);
let __VLS_561;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_562 = __VLS_asFunctionalComponent1(__VLS_561, new __VLS_561({
    ...{ 'onClick': {} },
    type: "primary",
    size: "small",
}));
const __VLS_563 = __VLS_562({
    ...{ 'onClick': {} },
    type: "primary",
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_562));
let __VLS_566;
const __VLS_567 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleUpdateStatus(2);
            // @ts-ignore
            [orderReturnApply, handleUpdateStatus,];
        } });
const { default: __VLS_568 } = __VLS_564.slots;
// @ts-ignore
[];
var __VLS_564;
var __VLS_565;
// @ts-ignore
[];
var __VLS_58;
// @ts-ignore
var __VLS_12 = __VLS_11;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
