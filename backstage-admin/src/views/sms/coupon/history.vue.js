/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { formatDate, formatDateTime } from '@/utils/datetime';
import { Search } from '@element-plus/icons-vue';
import { getCouponByIdAPI, getCouponHistoryListAPI } from '@/apis/coupon';
import { couponTypes } from '@/utils/constant';
import { useRoute } from 'vue-router';
// 获取路由
const route = useRoute();
// 优惠券详情
const coupon = ref({});
// 优惠券历史列表查询参数
const listQuery = ref({
    pageNum: 1,
    pageSize: 10,
});
// 优惠券历史列表数据
const list = ref([]);
// 总记录数
const total = ref(0);
// 表格数据加载状态
const listLoading = ref(false);
// 获取列表数据
const getList = async () => {
    listLoading.value = true;
    try {
        const response = await getCouponHistoryListAPI(listQuery.value);
        listLoading.value = false;
        list.value = response.data.list;
        total.value = response.data.total;
    }
    catch (error) {
        listLoading.value = false;
        console.error('获取优惠券历史列表失败:', error);
    }
};
// 使用状态选项
const useTypeOptions = ref([
    { label: "未使用", value: 0 },
    { label: "已使用", value: 1 },
    { label: "已过期", value: 2 }
]);
// 页面加载完成后获取数据
onMounted(async () => {
    // 获取优惠券详情
    const couponRes = await getCouponByIdAPI(Number(route.query.id));
    coupon.value = couponRes.data;
    // 设置优惠券ID用于查询历史记录
    listQuery.value.couponId = Number(route.query.id);
    getList();
});
// 重置搜索条件
const handleResetSearch = () => {
    listQuery.value = {
        pageNum: 1,
        pageSize: 10,
        couponId: Number(route.query.id)
    };
};
// 搜索列表
const handleSearchList = async () => {
    listQuery.value.pageNum = 1;
    await getList();
};
// 每页条数变化
const handleSizeChange = (val) => {
    listQuery.value.pageNum = 1;
    listQuery.value.pageSize = val;
    getList();
};
// 当前页变化
const handleCurrentChange = (val) => {
    listQuery.value.pageNum = val;
    getList();
};
// 优惠券类型过滤器
const formatType = (type) => {
    const found = couponTypes.find(option => option.value === type);
    return found ? found.label : '';
};
// 优惠券使用类型过滤器
const formatUseType = (useType) => {
    if (useType === 0) {
        return '全场通用';
    }
    else if (useType === 1) {
        return '指定分类';
    }
    else {
        return '指定商品';
    }
};
// 优惠券过期状态过滤器
const formatStatus = (endTime) => {
    if (!endTime)
        return '';
    const endTimeDate = new Date(endTime).getTime();
    const now = new Date().getTime();
    return endTimeDate > now ? '未过期' : '已过期';
};
// 获取类型过滤器
const formatGetType = (type) => {
    if (type === 1) {
        return '主动获取';
    }
    else {
        return '后台赠送';
    }
};
// 优惠券历史使用类型过滤器
const formatCouponHistoryUseType = (useType) => {
    if (useType === 0) {
        return '未使用';
    }
    else if (useType === 1) {
        return '已使用';
    }
    else {
        return '已过期';
    }
};
const __VLS_ctx = {
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "app-container" },
});
/** @type {__VLS_StyleScopedClasses['app-container']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-layout" },
});
/** @type {__VLS_StyleScopedClasses['table-layout']} */ ;
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({}));
const __VLS_2 = __VLS_1({}, ...__VLS_functionalComponentArgsRest(__VLS_1));
const { default: __VLS_5 } = __VLS_3.slots;
let __VLS_6;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_8 = __VLS_7({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_11 } = __VLS_9.slots;
var __VLS_9;
let __VLS_12;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_13 = __VLS_asFunctionalComponent1(__VLS_12, new __VLS_12({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_14 = __VLS_13({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_13));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_17 } = __VLS_15.slots;
var __VLS_15;
let __VLS_18;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_19 = __VLS_asFunctionalComponent1(__VLS_18, new __VLS_18({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_20 = __VLS_19({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_19));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_23 } = __VLS_21.slots;
var __VLS_21;
let __VLS_24;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_25 = __VLS_asFunctionalComponent1(__VLS_24, new __VLS_24({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_26 = __VLS_25({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_25));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_29 } = __VLS_27.slots;
var __VLS_27;
let __VLS_30;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_31 = __VLS_asFunctionalComponent1(__VLS_30, new __VLS_30({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_32 = __VLS_31({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_31));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_35 } = __VLS_33.slots;
var __VLS_33;
let __VLS_36;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_37 = __VLS_asFunctionalComponent1(__VLS_36, new __VLS_36({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_38 = __VLS_37({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_37));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_41 } = __VLS_39.slots;
var __VLS_39;
var __VLS_3;
let __VLS_42;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_43 = __VLS_asFunctionalComponent1(__VLS_42, new __VLS_42({}));
const __VLS_44 = __VLS_43({}, ...__VLS_functionalComponentArgsRest(__VLS_43));
const { default: __VLS_47 } = __VLS_45.slots;
let __VLS_48;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_49 = __VLS_asFunctionalComponent1(__VLS_48, new __VLS_48({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_50 = __VLS_49({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_49));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_53 } = __VLS_51.slots;
(__VLS_ctx.coupon.name);
// @ts-ignore
[coupon,];
var __VLS_51;
let __VLS_54;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_56 = __VLS_55({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_59 } = __VLS_57.slots;
(__VLS_ctx.formatType(__VLS_ctx.coupon.type));
// @ts-ignore
[coupon, formatType,];
var __VLS_57;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_62 = __VLS_61({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_65 } = __VLS_63.slots;
(__VLS_ctx.formatUseType(__VLS_ctx.coupon.useType));
// @ts-ignore
[coupon, formatUseType,];
var __VLS_63;
let __VLS_66;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_67 = __VLS_asFunctionalComponent1(__VLS_66, new __VLS_66({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_68 = __VLS_67({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_67));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_71 } = __VLS_69.slots;
(__VLS_ctx.coupon.minPoint);
// @ts-ignore
[coupon,];
var __VLS_69;
let __VLS_72;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_73 = __VLS_asFunctionalComponent1(__VLS_72, new __VLS_72({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_74 = __VLS_73({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_73));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_77 } = __VLS_75.slots;
(__VLS_ctx.coupon.amount);
// @ts-ignore
[coupon,];
var __VLS_75;
let __VLS_78;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_79 = __VLS_asFunctionalComponent1(__VLS_78, new __VLS_78({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_80 = __VLS_79({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_79));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_83 } = __VLS_81.slots;
(__VLS_ctx.formatStatus(__VLS_ctx.coupon.endTime));
// @ts-ignore
[coupon, formatStatus,];
var __VLS_81;
// @ts-ignore
[];
var __VLS_45;
let __VLS_84;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_85 = __VLS_asFunctionalComponent1(__VLS_84, new __VLS_84({}));
const __VLS_86 = __VLS_85({}, ...__VLS_functionalComponentArgsRest(__VLS_85));
const { default: __VLS_89 } = __VLS_87.slots;
let __VLS_90;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_91 = __VLS_asFunctionalComponent1(__VLS_90, new __VLS_90({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_92 = __VLS_91({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_91));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_95 } = __VLS_93.slots;
// @ts-ignore
[];
var __VLS_93;
let __VLS_96;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_97 = __VLS_asFunctionalComponent1(__VLS_96, new __VLS_96({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_98 = __VLS_97({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_97));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_101 } = __VLS_99.slots;
// @ts-ignore
[];
var __VLS_99;
let __VLS_102;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_103 = __VLS_asFunctionalComponent1(__VLS_102, new __VLS_102({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_104 = __VLS_103({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_103));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_107 } = __VLS_105.slots;
// @ts-ignore
[];
var __VLS_105;
let __VLS_108;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_109 = __VLS_asFunctionalComponent1(__VLS_108, new __VLS_108({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_110 = __VLS_109({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_109));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_113 } = __VLS_111.slots;
// @ts-ignore
[];
var __VLS_111;
let __VLS_114;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_115 = __VLS_asFunctionalComponent1(__VLS_114, new __VLS_114({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_116 = __VLS_115({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_115));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_119 } = __VLS_117.slots;
// @ts-ignore
[];
var __VLS_117;
let __VLS_120;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_121 = __VLS_asFunctionalComponent1(__VLS_120, new __VLS_120({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_122 = __VLS_121({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_121));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_125 } = __VLS_123.slots;
// @ts-ignore
[];
var __VLS_123;
// @ts-ignore
[];
var __VLS_87;
let __VLS_126;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_127 = __VLS_asFunctionalComponent1(__VLS_126, new __VLS_126({}));
const __VLS_128 = __VLS_127({}, ...__VLS_functionalComponentArgsRest(__VLS_127));
const { default: __VLS_131 } = __VLS_129.slots;
let __VLS_132;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_133 = __VLS_asFunctionalComponent1(__VLS_132, new __VLS_132({
    span: (4),
    ...{ class: "table-cell" },
    ...{ style: {} },
}));
const __VLS_134 = __VLS_133({
    span: (4),
    ...{ class: "table-cell" },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_133));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_137 } = __VLS_135.slots;
(__VLS_ctx.formatDate(__VLS_ctx.coupon.startTime));
(__VLS_ctx.formatDate(__VLS_ctx.coupon.endTime));
// @ts-ignore
[coupon, coupon, formatDate, formatDate,];
var __VLS_135;
let __VLS_138;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_139 = __VLS_asFunctionalComponent1(__VLS_138, new __VLS_138({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_140 = __VLS_139({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_139));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_143 } = __VLS_141.slots;
(__VLS_ctx.coupon.publishCount);
// @ts-ignore
[coupon,];
var __VLS_141;
let __VLS_144;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_145 = __VLS_asFunctionalComponent1(__VLS_144, new __VLS_144({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_146 = __VLS_145({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_145));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_149 } = __VLS_147.slots;
(__VLS_ctx.coupon.receiveCount);
// @ts-ignore
[coupon,];
var __VLS_147;
let __VLS_150;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_151 = __VLS_asFunctionalComponent1(__VLS_150, new __VLS_150({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_152 = __VLS_151({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_151));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_155 } = __VLS_153.slots;
(__VLS_ctx.coupon.publishCount - __VLS_ctx.coupon.receiveCount);
// @ts-ignore
[coupon, coupon,];
var __VLS_153;
let __VLS_156;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_157 = __VLS_asFunctionalComponent1(__VLS_156, new __VLS_156({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_158 = __VLS_157({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_157));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_161 } = __VLS_159.slots;
(__VLS_ctx.coupon.useCount);
// @ts-ignore
[coupon,];
var __VLS_159;
let __VLS_162;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_163 = __VLS_asFunctionalComponent1(__VLS_162, new __VLS_162({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_164 = __VLS_163({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_163));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_167 } = __VLS_165.slots;
(__VLS_ctx.coupon.publishCount - __VLS_ctx.coupon.useCount);
// @ts-ignore
[coupon, coupon,];
var __VLS_165;
// @ts-ignore
[];
var __VLS_129;
let __VLS_168;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_169 = __VLS_asFunctionalComponent1(__VLS_168, new __VLS_168({
    ...{ class: "filter-container" },
    shadow: "never",
}));
const __VLS_170 = __VLS_169({
    ...{ class: "filter-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_169));
/** @type {__VLS_StyleScopedClasses['filter-container']} */ ;
const { default: __VLS_173 } = __VLS_171.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
let __VLS_174;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_175 = __VLS_asFunctionalComponent1(__VLS_174, new __VLS_174({
    ...{ class: "el-icon-middle" },
}));
const __VLS_176 = __VLS_175({
    ...{ class: "el-icon-middle" },
}, ...__VLS_functionalComponentArgsRest(__VLS_175));
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_179 } = __VLS_177.slots;
let __VLS_180;
/** @ts-ignore @type { | typeof __VLS_components.Search} */
Search;
// @ts-ignore
const __VLS_181 = __VLS_asFunctionalComponent1(__VLS_180, new __VLS_180({}));
const __VLS_182 = __VLS_181({}, ...__VLS_functionalComponentArgsRest(__VLS_181));
// @ts-ignore
[];
var __VLS_177;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
let __VLS_185;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_186 = __VLS_asFunctionalComponent1(__VLS_185, new __VLS_185({
    ...{ 'onClick': {} },
    ...{ style: {} },
    type: "primary",
}));
const __VLS_187 = __VLS_186({
    ...{ 'onClick': {} },
    ...{ style: {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_186));
let __VLS_190;
const __VLS_191 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleSearchList();
            // @ts-ignore
            [handleSearchList,];
        } });
const { default: __VLS_192 } = __VLS_188.slots;
// @ts-ignore
[];
var __VLS_188;
var __VLS_189;
let __VLS_193;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_194 = __VLS_asFunctionalComponent1(__VLS_193, new __VLS_193({
    ...{ 'onClick': {} },
    ...{ style: {} },
}));
const __VLS_195 = __VLS_194({
    ...{ 'onClick': {} },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_194));
let __VLS_198;
const __VLS_199 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleResetSearch();
            // @ts-ignore
            [handleResetSearch,];
        } });
const { default: __VLS_200 } = __VLS_196.slots;
// @ts-ignore
[];
var __VLS_196;
var __VLS_197;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_201;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_202 = __VLS_asFunctionalComponent1(__VLS_201, new __VLS_201({
    inline: (true),
    model: (__VLS_ctx.listQuery),
    labelWidth: "140px",
}));
const __VLS_203 = __VLS_202({
    inline: (true),
    model: (__VLS_ctx.listQuery),
    labelWidth: "140px",
}, ...__VLS_functionalComponentArgsRest(__VLS_202));
const { default: __VLS_206 } = __VLS_204.slots;
let __VLS_207;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_208 = __VLS_asFunctionalComponent1(__VLS_207, new __VLS_207({
    label: "使用状态：",
}));
const __VLS_209 = __VLS_208({
    label: "使用状态：",
}, ...__VLS_functionalComponentArgsRest(__VLS_208));
const { default: __VLS_212 } = __VLS_210.slots;
let __VLS_213;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_214 = __VLS_asFunctionalComponent1(__VLS_213, new __VLS_213({
    modelValue: (__VLS_ctx.listQuery.useStatus),
    placeholder: "全部",
    clearable: true,
    ...{ style: {} },
}));
const __VLS_215 = __VLS_214({
    modelValue: (__VLS_ctx.listQuery.useStatus),
    placeholder: "全部",
    clearable: true,
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_214));
const { default: __VLS_218 } = __VLS_216.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.useTypeOptions))) {
    let __VLS_219;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_220 = __VLS_asFunctionalComponent1(__VLS_219, new __VLS_219({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_221 = __VLS_220({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_220));
    // @ts-ignore
    [listQuery, listQuery, useTypeOptions,];
}
// @ts-ignore
[];
var __VLS_216;
// @ts-ignore
[];
var __VLS_210;
let __VLS_224;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_225 = __VLS_asFunctionalComponent1(__VLS_224, new __VLS_224({
    label: "订单编号：",
}));
const __VLS_226 = __VLS_225({
    label: "订单编号：",
}, ...__VLS_functionalComponentArgsRest(__VLS_225));
const { default: __VLS_229 } = __VLS_227.slots;
let __VLS_230;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_231 = __VLS_asFunctionalComponent1(__VLS_230, new __VLS_230({
    modelValue: (__VLS_ctx.listQuery.orderSn),
    ...{ class: "input-width" },
    placeholder: "订单编号",
}));
const __VLS_232 = __VLS_231({
    modelValue: (__VLS_ctx.listQuery.orderSn),
    ...{ class: "input-width" },
    placeholder: "订单编号",
}, ...__VLS_functionalComponentArgsRest(__VLS_231));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery,];
var __VLS_227;
// @ts-ignore
[];
var __VLS_204;
// @ts-ignore
[];
var __VLS_171;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_235;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_236 = __VLS_asFunctionalComponent1(__VLS_235, new __VLS_235({
    ref: "couponHistoryTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_237 = __VLS_236({
    ref: "couponHistoryTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_236));
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_240 = {};
const { default: __VLS_242 } = __VLS_238.slots;
let __VLS_243;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_244 = __VLS_asFunctionalComponent1(__VLS_243, new __VLS_243({
    label: "优惠码",
    width: "160",
    align: "center",
}));
const __VLS_245 = __VLS_244({
    label: "优惠码",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_244));
const { default: __VLS_248 } = __VLS_246.slots;
{
    const { default: __VLS_249 } = __VLS_246.slots;
    const [scope] = __VLS_vSlot(__VLS_249);
    (scope.row.couponCode);
    // @ts-ignore
    [list, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_246;
let __VLS_250;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_251 = __VLS_asFunctionalComponent1(__VLS_250, new __VLS_250({
    label: "领取会员",
    width: "140",
    align: "center",
}));
const __VLS_252 = __VLS_251({
    label: "领取会员",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_251));
const { default: __VLS_255 } = __VLS_253.slots;
{
    const { default: __VLS_256 } = __VLS_253.slots;
    const [scope] = __VLS_vSlot(__VLS_256);
    (scope.row.memberNickname);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_253;
let __VLS_257;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_258 = __VLS_asFunctionalComponent1(__VLS_257, new __VLS_257({
    label: "领取方式",
    width: "100",
    align: "center",
}));
const __VLS_259 = __VLS_258({
    label: "领取方式",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_258));
const { default: __VLS_262 } = __VLS_260.slots;
{
    const { default: __VLS_263 } = __VLS_260.slots;
    const [scope] = __VLS_vSlot(__VLS_263);
    (__VLS_ctx.formatGetType(scope.row.getType));
    // @ts-ignore
    [formatGetType,];
}
// @ts-ignore
[];
var __VLS_260;
let __VLS_264;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_265 = __VLS_asFunctionalComponent1(__VLS_264, new __VLS_264({
    label: "领取时间",
    width: "160",
    align: "center",
}));
const __VLS_266 = __VLS_265({
    label: "领取时间",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_265));
const { default: __VLS_269 } = __VLS_267.slots;
{
    const { default: __VLS_270 } = __VLS_267.slots;
    const [scope] = __VLS_vSlot(__VLS_270);
    (__VLS_ctx.formatDateTime(scope.row.createTime));
    // @ts-ignore
    [formatDateTime,];
}
// @ts-ignore
[];
var __VLS_267;
let __VLS_271;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_272 = __VLS_asFunctionalComponent1(__VLS_271, new __VLS_271({
    label: "当前状态",
    width: "140",
    align: "center",
}));
const __VLS_273 = __VLS_272({
    label: "当前状态",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_272));
const { default: __VLS_276 } = __VLS_274.slots;
{
    const { default: __VLS_277 } = __VLS_274.slots;
    const [scope] = __VLS_vSlot(__VLS_277);
    (__VLS_ctx.formatCouponHistoryUseType(scope.row.useStatus));
    // @ts-ignore
    [formatCouponHistoryUseType,];
}
// @ts-ignore
[];
var __VLS_274;
let __VLS_278;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_279 = __VLS_asFunctionalComponent1(__VLS_278, new __VLS_278({
    label: "使用时间",
    width: "160",
    align: "center",
}));
const __VLS_280 = __VLS_279({
    label: "使用时间",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_279));
const { default: __VLS_283 } = __VLS_281.slots;
{
    const { default: __VLS_284 } = __VLS_281.slots;
    const [scope] = __VLS_vSlot(__VLS_284);
    (__VLS_ctx.formatDateTime(scope.row.useTime));
    // @ts-ignore
    [formatDateTime,];
}
// @ts-ignore
[];
var __VLS_281;
let __VLS_285;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_286 = __VLS_asFunctionalComponent1(__VLS_285, new __VLS_285({
    label: "订单编号",
    align: "center",
}));
const __VLS_287 = __VLS_286({
    label: "订单编号",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_286));
const { default: __VLS_290 } = __VLS_288.slots;
{
    const { default: __VLS_291 } = __VLS_288.slots;
    const [scope] = __VLS_vSlot(__VLS_291);
    (scope.row.orderSn === null ? 'N/A' : scope.row.orderSn);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_288;
// @ts-ignore
[];
var __VLS_238;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_292;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_293 = __VLS_asFunctionalComponent1(__VLS_292, new __VLS_292({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}));
const __VLS_294 = __VLS_293({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_293));
let __VLS_297;
const __VLS_298 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_299 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_295;
var __VLS_296;
// @ts-ignore
var __VLS_241 = __VLS_240;
// @ts-ignore
[listQuery, listQuery, total, handleSizeChange, handleCurrentChange,];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
