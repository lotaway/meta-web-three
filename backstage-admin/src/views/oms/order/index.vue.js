/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { Search, Tickets } from '@element-plus/icons-vue';
import { getOrderListAPI, orderUpdateCloseAPI, orderDeleteByIdsAPI } from '@/apis/order';
import LogisticsDialog from '@/views/oms/order/components/logisticsDialog.vue';
import { formatDateTime } from '@/utils/datetime';
import { useOrderStore } from '@/stores/order';
// 获取路由对象
const router = useRouter();
// 获取订单存储store
const orderStore = useOrderStore();
// 订单列表查询参数
const listQuery = ref({
    pageNum: 1,
    pageSize: 10
});
// 订单列表数据
const list = ref([]);
// 表格数据加载进度条
const listLoading = ref(true);
// 分页组件参数
const total = ref(0);
// 获取订单列表数据
const getList = async () => {
    listLoading.value = true;
    try {
        const response = await getOrderListAPI(listQuery.value);
        listLoading.value = false;
        list.value = response.data.list;
        total.value = response.data.total;
    }
    catch (error) {
        listLoading.value = false;
        console.error('获取订单列表失败:', error);
    }
};
// 组件挂载后初始化列表信息
onMounted(() => {
    getList();
});
// 表格中被选中的行
const multipleSelection = ref([]);
// 批量操作类型
const operateType = ref();
// 关闭订单对话框相关数据
const closeOrderData = ref({
    dialogVisible: false,
    content: '',
    orderIds: []
});
// 物流对话框可见性
const logisticsDialogVisible = ref(false);
// 订单状态选项
const statusOptions = [
    {
        label: '待付款',
        value: 0
    },
    {
        label: '待发货',
        value: 1
    },
    {
        label: '已发货',
        value: 2
    },
    {
        label: '已完成',
        value: 3
    },
    {
        label: '已关闭',
        value: 4
    }
];
// 订单类型选项
const orderTypeOptions = [
    {
        label: '正常订单',
        value: 0
    },
    {
        label: '秒杀订单',
        value: 1
    }
];
// 订单来源选项
const sourceTypeOptions = [
    {
        label: 'PC订单',
        value: 0
    },
    {
        label: 'APP订单',
        value: 1
    }
];
// 批量操作选项
const operateOptions = [
    {
        label: "批量发货",
        value: 1
    },
    {
        label: "关闭订单",
        value: 2
    },
    {
        label: "删除订单",
        value: 3
    }
];
// 格式化支付方式
const formatPayType = (value) => {
    if (value === 1) {
        return '支付宝';
    }
    else if (value === 2) {
        return '微信';
    }
    else {
        return '未支付';
    }
};
// 格式化订单来源
const formatSourceType = (value) => {
    if (value === 1) {
        return 'APP订单';
    }
    else {
        return 'PC订单';
    }
};
// 格式化订单状态
const formatStatus = (value) => {
    if (value === 1) {
        return '待发货';
    }
    else if (value === 2) {
        return '已发货';
    }
    else if (value === 3) {
        return '已完成';
    }
    else if (value === 4) {
        return '已关闭';
    }
    else if (value === 5) {
        return '无效订单';
    }
    else {
        return '待付款';
    }
};
// 处理重置搜索
const handleResetSearch = () => {
    listQuery.value = { pageNum: 1, pageSize: 10 };
};
// 处理搜索列表
const handleSearchList = () => {
    listQuery.value.pageNum = 1;
    getList();
};
// 处理表格选中状态变化
const handleSelectionChange = (val) => {
    multipleSelection.value = val;
};
// 处理查看订单
const handleViewOrder = (index, row) => {
    router.push({ path: '/oms/orderDetail', query: { id: row.id } });
};
// 处理关闭订单
const handleCloseOrder = (index, row) => {
    closeOrderData.value.dialogVisible = true;
    closeOrderData.value.orderIds = [row.id];
};
// 处理订单发货
const handleDeliveryOrder = (index, row) => {
    orderStore.setDeliverOrderList([row]);
    router.push({ path: '/oms/deliverOrderList' });
};
// 处理查看物流
const handleViewLogistics = (index, row) => {
    logisticsDialogVisible.value = true;
    console.log(index, row);
};
// 处理删除订单
const handleDeleteOrder = async (index, row) => {
    const ids = [row.id];
    await deleteOrderFn(ids);
};
// 处理批量操作
const handleBatchOperate = async () => {
    if (!multipleSelection.value || multipleSelection.value.length < 1) {
        ElMessage({
            message: '请选择要操作的订单',
            type: 'warning',
            duration: 1000
        });
        return;
    }
    if (operateType.value === 1) {
        // 批量发货
        const listItems = multipleSelection.value.filter(item => item.status === 1);
        if (!listItems || listItems.length < 1) {
            ElMessage({
                message: '选中订单中没有可以发货的订单',
                type: 'warning',
                duration: 1000
            });
            return;
        }
        orderStore.setDeliverOrderList(listItems);
        router.push({ path: '/oms/deliverOrderList' });
    }
    else if (operateType.value === 2) {
        // 关闭订单
        closeOrderData.value.orderIds = multipleSelection.value.filter(item => item.status === 0)
            .map(item => item.id);
        closeOrderData.value.dialogVisible = true;
    }
    else if (operateType.value === 3) {
        // 删除订单
        const ids = multipleSelection.value.filter(item => item.status === 4)
            .map(item => item.id);
        await deleteOrderFn(ids);
    }
};
// 处理每页条数变化
const handleSizeChange = (val) => {
    listQuery.value.pageNum = 1;
    listQuery.value.pageSize = val;
    getList();
};
// 处理当前页变化
const handleCurrentChange = (val) => {
    listQuery.value.pageNum = val;
    getList();
};
// 处理确认关闭订单
const handleCloseOrderConfirm = async () => {
    if (!closeOrderData.value.content) {
        ElMessage({
            message: '操作备注不能为空',
            type: 'warning',
            duration: 1000
        });
        return;
    }
    const orderIds = closeOrderData.value.orderIds.join(',');
    await orderUpdateCloseAPI({ ids: orderIds, note: closeOrderData.value.content });
    closeOrderData.value.orderIds = [];
    closeOrderData.value.dialogVisible = false;
    getList();
    ElMessage({
        message: '修改成功',
        type: 'success',
        duration: 1000
    });
};
// 删除订单函数
const deleteOrderFn = async (ids) => {
    await ElMessageBox.confirm('是否要进行该删除操作?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    await orderDeleteByIdsAPI({ ids: ids.join(',') });
    ElMessage({
        message: '删除成功！',
        type: 'success',
        duration: 1000
    });
    getList();
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
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    ...{ class: "filter-container" },
    shadow: "never",
}));
const __VLS_2 = __VLS_1({
    ...{ class: "filter-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
/** @type {__VLS_StyleScopedClasses['filter-container']} */ ;
const { default: __VLS_5 } = __VLS_3.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
let __VLS_6;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({
    ...{ class: "el-icon-middle" },
}));
const __VLS_8 = __VLS_7({
    ...{ class: "el-icon-middle" },
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_11 } = __VLS_9.slots;
let __VLS_12;
/** @ts-ignore @type { | typeof __VLS_components.Search} */
Search;
// @ts-ignore
const __VLS_13 = __VLS_asFunctionalComponent1(__VLS_12, new __VLS_12({}));
const __VLS_14 = __VLS_13({}, ...__VLS_functionalComponentArgsRest(__VLS_13));
var __VLS_9;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
let __VLS_17;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_18 = __VLS_asFunctionalComponent1(__VLS_17, new __VLS_17({
    ...{ 'onClick': {} },
    ...{ style: {} },
    type: "primary",
}));
const __VLS_19 = __VLS_18({
    ...{ 'onClick': {} },
    ...{ style: {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_18));
let __VLS_22;
const __VLS_23 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleSearchList();
            // @ts-ignore
            [handleSearchList,];
        } });
const { default: __VLS_24 } = __VLS_20.slots;
// @ts-ignore
[];
var __VLS_20;
var __VLS_21;
let __VLS_25;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_26 = __VLS_asFunctionalComponent1(__VLS_25, new __VLS_25({
    ...{ 'onClick': {} },
    ...{ style: {} },
}));
const __VLS_27 = __VLS_26({
    ...{ 'onClick': {} },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_26));
let __VLS_30;
const __VLS_31 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleResetSearch();
            // @ts-ignore
            [handleResetSearch,];
        } });
const { default: __VLS_32 } = __VLS_28.slots;
// @ts-ignore
[];
var __VLS_28;
var __VLS_29;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_33;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_34 = __VLS_asFunctionalComponent1(__VLS_33, new __VLS_33({
    inline: (true),
    model: (__VLS_ctx.listQuery),
    labelWidth: "140px",
}));
const __VLS_35 = __VLS_34({
    inline: (true),
    model: (__VLS_ctx.listQuery),
    labelWidth: "140px",
}, ...__VLS_functionalComponentArgsRest(__VLS_34));
const { default: __VLS_38 } = __VLS_36.slots;
let __VLS_39;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_40 = __VLS_asFunctionalComponent1(__VLS_39, new __VLS_39({
    label: "输入搜索：",
}));
const __VLS_41 = __VLS_40({
    label: "输入搜索：",
}, ...__VLS_functionalComponentArgsRest(__VLS_40));
const { default: __VLS_44 } = __VLS_42.slots;
let __VLS_45;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_46 = __VLS_asFunctionalComponent1(__VLS_45, new __VLS_45({
    modelValue: (__VLS_ctx.listQuery.orderSn),
    ...{ class: "input-width" },
    placeholder: "订单编号",
}));
const __VLS_47 = __VLS_46({
    modelValue: (__VLS_ctx.listQuery.orderSn),
    ...{ class: "input-width" },
    placeholder: "订单编号",
}, ...__VLS_functionalComponentArgsRest(__VLS_46));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery, listQuery,];
var __VLS_42;
let __VLS_50;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_51 = __VLS_asFunctionalComponent1(__VLS_50, new __VLS_50({
    label: "收货人：",
}));
const __VLS_52 = __VLS_51({
    label: "收货人：",
}, ...__VLS_functionalComponentArgsRest(__VLS_51));
const { default: __VLS_55 } = __VLS_53.slots;
let __VLS_56;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
    modelValue: (__VLS_ctx.listQuery.receiverKeyword),
    ...{ class: "input-width" },
    placeholder: "收货人姓名/手机号码",
}));
const __VLS_58 = __VLS_57({
    modelValue: (__VLS_ctx.listQuery.receiverKeyword),
    ...{ class: "input-width" },
    placeholder: "收货人姓名/手机号码",
}, ...__VLS_functionalComponentArgsRest(__VLS_57));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery,];
var __VLS_53;
let __VLS_61;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_62 = __VLS_asFunctionalComponent1(__VLS_61, new __VLS_61({
    label: "提交时间：",
}));
const __VLS_63 = __VLS_62({
    label: "提交时间：",
}, ...__VLS_functionalComponentArgsRest(__VLS_62));
const { default: __VLS_66 } = __VLS_64.slots;
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    ...{ class: "input-width" },
    modelValue: (__VLS_ctx.listQuery.createTime),
    valueFormat: "yyyy-MM-dd",
    type: "date",
    placeholder: "请选择时间",
}));
const __VLS_69 = __VLS_68({
    ...{ class: "input-width" },
    modelValue: (__VLS_ctx.listQuery.createTime),
    valueFormat: "yyyy-MM-dd",
    type: "date",
    placeholder: "请选择时间",
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery,];
var __VLS_64;
let __VLS_72;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_73 = __VLS_asFunctionalComponent1(__VLS_72, new __VLS_72({
    label: "订单状态：",
}));
const __VLS_74 = __VLS_73({
    label: "订单状态：",
}, ...__VLS_functionalComponentArgsRest(__VLS_73));
const { default: __VLS_77 } = __VLS_75.slots;
let __VLS_78;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_79 = __VLS_asFunctionalComponent1(__VLS_78, new __VLS_78({
    modelValue: (__VLS_ctx.listQuery.status),
    ...{ class: "input-width" },
    placeholder: "全部",
    clearable: true,
}));
const __VLS_80 = __VLS_79({
    modelValue: (__VLS_ctx.listQuery.status),
    ...{ class: "input-width" },
    placeholder: "全部",
    clearable: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_79));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_83 } = __VLS_81.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.statusOptions))) {
    let __VLS_84;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_85 = __VLS_asFunctionalComponent1(__VLS_84, new __VLS_84({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_86 = __VLS_85({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_85));
    // @ts-ignore
    [listQuery, statusOptions,];
}
// @ts-ignore
[];
var __VLS_81;
// @ts-ignore
[];
var __VLS_75;
let __VLS_89;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_90 = __VLS_asFunctionalComponent1(__VLS_89, new __VLS_89({
    label: "订单分类：",
}));
const __VLS_91 = __VLS_90({
    label: "订单分类：",
}, ...__VLS_functionalComponentArgsRest(__VLS_90));
const { default: __VLS_94 } = __VLS_92.slots;
let __VLS_95;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_96 = __VLS_asFunctionalComponent1(__VLS_95, new __VLS_95({
    modelValue: (__VLS_ctx.listQuery.orderType),
    ...{ class: "input-width" },
    placeholder: "全部",
    clearable: true,
}));
const __VLS_97 = __VLS_96({
    modelValue: (__VLS_ctx.listQuery.orderType),
    ...{ class: "input-width" },
    placeholder: "全部",
    clearable: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_96));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_100 } = __VLS_98.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.orderTypeOptions))) {
    let __VLS_101;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_102 = __VLS_asFunctionalComponent1(__VLS_101, new __VLS_101({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_103 = __VLS_102({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_102));
    // @ts-ignore
    [listQuery, orderTypeOptions,];
}
// @ts-ignore
[];
var __VLS_98;
// @ts-ignore
[];
var __VLS_92;
let __VLS_106;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_107 = __VLS_asFunctionalComponent1(__VLS_106, new __VLS_106({
    label: "订单来源：",
}));
const __VLS_108 = __VLS_107({
    label: "订单来源：",
}, ...__VLS_functionalComponentArgsRest(__VLS_107));
const { default: __VLS_111 } = __VLS_109.slots;
let __VLS_112;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_113 = __VLS_asFunctionalComponent1(__VLS_112, new __VLS_112({
    modelValue: (__VLS_ctx.listQuery.sourceType),
    ...{ class: "input-width" },
    placeholder: "全部",
    clearable: true,
}));
const __VLS_114 = __VLS_113({
    modelValue: (__VLS_ctx.listQuery.sourceType),
    ...{ class: "input-width" },
    placeholder: "全部",
    clearable: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_113));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_117 } = __VLS_115.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.sourceTypeOptions))) {
    let __VLS_118;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_119 = __VLS_asFunctionalComponent1(__VLS_118, new __VLS_118({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_120 = __VLS_119({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_119));
    // @ts-ignore
    [listQuery, sourceTypeOptions,];
}
// @ts-ignore
[];
var __VLS_115;
// @ts-ignore
[];
var __VLS_109;
// @ts-ignore
[];
var __VLS_36;
// @ts-ignore
[];
var __VLS_3;
let __VLS_123;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_124 = __VLS_asFunctionalComponent1(__VLS_123, new __VLS_123({
    ...{ class: "operate-container" },
    shadow: "never",
}));
const __VLS_125 = __VLS_124({
    ...{ class: "operate-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_124));
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
const { default: __VLS_128 } = __VLS_126.slots;
let __VLS_129;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_130 = __VLS_asFunctionalComponent1(__VLS_129, new __VLS_129({
    ...{ class: "el-icon-middle" },
}));
const __VLS_131 = __VLS_130({
    ...{ class: "el-icon-middle" },
}, ...__VLS_functionalComponentArgsRest(__VLS_130));
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_134 } = __VLS_132.slots;
let __VLS_135;
/** @ts-ignore @type { | typeof __VLS_components.Tickets} */
Tickets;
// @ts-ignore
const __VLS_136 = __VLS_asFunctionalComponent1(__VLS_135, new __VLS_135({}));
const __VLS_137 = __VLS_136({}, ...__VLS_functionalComponentArgsRest(__VLS_136));
// @ts-ignore
[];
var __VLS_132;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
// @ts-ignore
[];
var __VLS_126;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_140;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_141 = __VLS_asFunctionalComponent1(__VLS_140, new __VLS_140({
    ...{ 'onSelectionChange': {} },
    ref: "orderTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_142 = __VLS_141({
    ...{ 'onSelectionChange': {} },
    ref: "orderTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_141));
let __VLS_145;
const __VLS_146 = ({ selectionChange: {} },
    { onSelectionChange: (__VLS_ctx.handleSelectionChange) });
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_147 = {};
const { default: __VLS_149 } = __VLS_143.slots;
let __VLS_150;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_151 = __VLS_asFunctionalComponent1(__VLS_150, new __VLS_150({
    type: "selection",
    width: "60",
    align: "center",
}));
const __VLS_152 = __VLS_151({
    type: "selection",
    width: "60",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_151));
let __VLS_155;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_156 = __VLS_asFunctionalComponent1(__VLS_155, new __VLS_155({
    label: "编号",
    width: "80",
    align: "center",
}));
const __VLS_157 = __VLS_156({
    label: "编号",
    width: "80",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_156));
const { default: __VLS_160 } = __VLS_158.slots;
{
    const { default: __VLS_161 } = __VLS_158.slots;
    const [scope] = __VLS_vSlot(__VLS_161);
    (scope.row.id);
    // @ts-ignore
    [list, handleSelectionChange, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_158;
let __VLS_162;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_163 = __VLS_asFunctionalComponent1(__VLS_162, new __VLS_162({
    label: "订单编号",
    width: "180",
    align: "center",
}));
const __VLS_164 = __VLS_163({
    label: "订单编号",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_163));
const { default: __VLS_167 } = __VLS_165.slots;
{
    const { default: __VLS_168 } = __VLS_165.slots;
    const [scope] = __VLS_vSlot(__VLS_168);
    (scope.row.orderSn);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_165;
let __VLS_169;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_170 = __VLS_asFunctionalComponent1(__VLS_169, new __VLS_169({
    label: "提交时间",
    width: "180",
    align: "center",
}));
const __VLS_171 = __VLS_170({
    label: "提交时间",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_170));
const { default: __VLS_174 } = __VLS_172.slots;
{
    const { default: __VLS_175 } = __VLS_172.slots;
    const [scope] = __VLS_vSlot(__VLS_175);
    (__VLS_ctx.formatDateTime(scope.row.createTime));
    // @ts-ignore
    [formatDateTime,];
}
// @ts-ignore
[];
var __VLS_172;
let __VLS_176;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_177 = __VLS_asFunctionalComponent1(__VLS_176, new __VLS_176({
    label: "用户账号",
    align: "center",
}));
const __VLS_178 = __VLS_177({
    label: "用户账号",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_177));
const { default: __VLS_181 } = __VLS_179.slots;
{
    const { default: __VLS_182 } = __VLS_179.slots;
    const [scope] = __VLS_vSlot(__VLS_182);
    (scope.row.memberUsername);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_179;
let __VLS_183;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_184 = __VLS_asFunctionalComponent1(__VLS_183, new __VLS_183({
    label: "订单金额",
    width: "120",
    align: "center",
}));
const __VLS_185 = __VLS_184({
    label: "订单金额",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_184));
const { default: __VLS_188 } = __VLS_186.slots;
{
    const { default: __VLS_189 } = __VLS_186.slots;
    const [scope] = __VLS_vSlot(__VLS_189);
    (scope.row.totalAmount);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_186;
let __VLS_190;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_191 = __VLS_asFunctionalComponent1(__VLS_190, new __VLS_190({
    label: "支付方式",
    width: "120",
    align: "center",
}));
const __VLS_192 = __VLS_191({
    label: "支付方式",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_191));
const { default: __VLS_195 } = __VLS_193.slots;
{
    const { default: __VLS_196 } = __VLS_193.slots;
    const [scope] = __VLS_vSlot(__VLS_196);
    (__VLS_ctx.formatPayType(scope.row.payType));
    // @ts-ignore
    [formatPayType,];
}
// @ts-ignore
[];
var __VLS_193;
let __VLS_197;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_198 = __VLS_asFunctionalComponent1(__VLS_197, new __VLS_197({
    label: "订单来源",
    width: "120",
    align: "center",
}));
const __VLS_199 = __VLS_198({
    label: "订单来源",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_198));
const { default: __VLS_202 } = __VLS_200.slots;
{
    const { default: __VLS_203 } = __VLS_200.slots;
    const [scope] = __VLS_vSlot(__VLS_203);
    (__VLS_ctx.formatSourceType(scope.row.sourceType));
    // @ts-ignore
    [formatSourceType,];
}
// @ts-ignore
[];
var __VLS_200;
let __VLS_204;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_205 = __VLS_asFunctionalComponent1(__VLS_204, new __VLS_204({
    label: "订单状态",
    width: "120",
    align: "center",
}));
const __VLS_206 = __VLS_205({
    label: "订单状态",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_205));
const { default: __VLS_209 } = __VLS_207.slots;
{
    const { default: __VLS_210 } = __VLS_207.slots;
    const [scope] = __VLS_vSlot(__VLS_210);
    (__VLS_ctx.formatStatus(scope.row.status));
    // @ts-ignore
    [formatStatus,];
}
// @ts-ignore
[];
var __VLS_207;
let __VLS_211;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_212 = __VLS_asFunctionalComponent1(__VLS_211, new __VLS_211({
    label: "操作",
    width: "200",
    align: "center",
}));
const __VLS_213 = __VLS_212({
    label: "操作",
    width: "200",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_212));
const { default: __VLS_216 } = __VLS_214.slots;
{
    const { default: __VLS_217 } = __VLS_214.slots;
    const [scope] = __VLS_vSlot(__VLS_217);
    let __VLS_218;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_219 = __VLS_asFunctionalComponent1(__VLS_218, new __VLS_218({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_220 = __VLS_219({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_219));
    let __VLS_223;
    const __VLS_224 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleViewOrder(scope.$index, scope.row);
                // @ts-ignore
                [handleViewOrder,];
            } });
    const { default: __VLS_225 } = __VLS_221.slots;
    // @ts-ignore
    [];
    var __VLS_221;
    var __VLS_222;
    let __VLS_226;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_227 = __VLS_asFunctionalComponent1(__VLS_226, new __VLS_226({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_228 = __VLS_227({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_227));
    let __VLS_231;
    const __VLS_232 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleCloseOrder(scope.$index, scope.row);
                // @ts-ignore
                [handleCloseOrder,];
            } });
    __VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (scope.row.status === 0) }, null, null);
    const { default: __VLS_233 } = __VLS_229.slots;
    // @ts-ignore
    [];
    var __VLS_229;
    var __VLS_230;
    let __VLS_234;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_235 = __VLS_asFunctionalComponent1(__VLS_234, new __VLS_234({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_236 = __VLS_235({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_235));
    let __VLS_239;
    const __VLS_240 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDeliveryOrder(scope.$index, scope.row);
                // @ts-ignore
                [handleDeliveryOrder,];
            } });
    __VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (scope.row.status === 1) }, null, null);
    const { default: __VLS_241 } = __VLS_237.slots;
    // @ts-ignore
    [];
    var __VLS_237;
    var __VLS_238;
    let __VLS_242;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_243 = __VLS_asFunctionalComponent1(__VLS_242, new __VLS_242({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_244 = __VLS_243({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_243));
    let __VLS_247;
    const __VLS_248 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleViewLogistics(scope.$index, scope.row);
                // @ts-ignore
                [handleViewLogistics,];
            } });
    __VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (scope.row.status === 2 || scope.row.status === 3) }, null, null);
    const { default: __VLS_249 } = __VLS_245.slots;
    // @ts-ignore
    [];
    var __VLS_245;
    var __VLS_246;
    let __VLS_250;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_251 = __VLS_asFunctionalComponent1(__VLS_250, new __VLS_250({
        ...{ 'onClick': {} },
        size: "small",
        type: "danger",
    }));
    const __VLS_252 = __VLS_251({
        ...{ 'onClick': {} },
        size: "small",
        type: "danger",
    }, ...__VLS_functionalComponentArgsRest(__VLS_251));
    let __VLS_255;
    const __VLS_256 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDeleteOrder(scope.$index, scope.row);
                // @ts-ignore
                [handleDeleteOrder,];
            } });
    __VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (scope.row.status === 4) }, null, null);
    const { default: __VLS_257 } = __VLS_253.slots;
    // @ts-ignore
    [];
    var __VLS_253;
    var __VLS_254;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_214;
// @ts-ignore
[];
var __VLS_143;
var __VLS_144;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "batch-operate-container" },
});
/** @type {__VLS_StyleScopedClasses['batch-operate-container']} */ ;
let __VLS_258;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_259 = __VLS_asFunctionalComponent1(__VLS_258, new __VLS_258({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}));
const __VLS_260 = __VLS_259({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}, ...__VLS_functionalComponentArgsRest(__VLS_259));
const { default: __VLS_263 } = __VLS_261.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.operateOptions))) {
    let __VLS_264;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_265 = __VLS_asFunctionalComponent1(__VLS_264, new __VLS_264({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_266 = __VLS_265({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_265));
    // @ts-ignore
    [operateType, operateOptions,];
}
// @ts-ignore
[];
var __VLS_261;
let __VLS_269;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_270 = __VLS_asFunctionalComponent1(__VLS_269, new __VLS_269({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}));
const __VLS_271 = __VLS_270({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_270));
let __VLS_274;
const __VLS_275 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleBatchOperate();
            // @ts-ignore
            [handleBatchOperate,];
        } });
/** @type {__VLS_StyleScopedClasses['search-button']} */ ;
const { default: __VLS_276 } = __VLS_272.slots;
// @ts-ignore
[];
var __VLS_272;
var __VLS_273;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_277;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_278 = __VLS_asFunctionalComponent1(__VLS_277, new __VLS_277({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}));
const __VLS_279 = __VLS_278({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_278));
let __VLS_282;
const __VLS_283 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_284 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_280;
var __VLS_281;
let __VLS_285;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_286 = __VLS_asFunctionalComponent1(__VLS_285, new __VLS_285({
    title: "关闭订单",
    modelValue: (__VLS_ctx.closeOrderData.dialogVisible),
    width: "30%",
}));
const __VLS_287 = __VLS_286({
    title: "关闭订单",
    modelValue: (__VLS_ctx.closeOrderData.dialogVisible),
    width: "30%",
}, ...__VLS_functionalComponentArgsRest(__VLS_286));
const { default: __VLS_290 } = __VLS_288.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
});
let __VLS_291;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_292 = __VLS_asFunctionalComponent1(__VLS_291, new __VLS_291({
    ...{ style: {} },
    type: "textarea",
    rows: (5),
    placeholder: "请输入内容",
    modelValue: (__VLS_ctx.closeOrderData.content),
}));
const __VLS_293 = __VLS_292({
    ...{ style: {} },
    type: "textarea",
    rows: (5),
    placeholder: "请输入内容",
    modelValue: (__VLS_ctx.closeOrderData.content),
}, ...__VLS_functionalComponentArgsRest(__VLS_292));
{
    const { footer: __VLS_296 } = __VLS_288.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_297;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_298 = __VLS_asFunctionalComponent1(__VLS_297, new __VLS_297({
        ...{ 'onClick': {} },
    }));
    const __VLS_299 = __VLS_298({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_298));
    let __VLS_302;
    const __VLS_303 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.closeOrderData.dialogVisible = false;
                // @ts-ignore
                [listQuery, listQuery, total, handleSizeChange, handleCurrentChange, closeOrderData, closeOrderData, closeOrderData,];
            } });
    const { default: __VLS_304 } = __VLS_300.slots;
    // @ts-ignore
    [];
    var __VLS_300;
    var __VLS_301;
    let __VLS_305;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_306 = __VLS_asFunctionalComponent1(__VLS_305, new __VLS_305({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_307 = __VLS_306({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_306));
    let __VLS_310;
    const __VLS_311 = ({ click: {} },
        { onClick: (__VLS_ctx.handleCloseOrderConfirm) });
    const { default: __VLS_312 } = __VLS_308.slots;
    // @ts-ignore
    [handleCloseOrderConfirm,];
    var __VLS_308;
    var __VLS_309;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_288;
const __VLS_313 = LogisticsDialog || LogisticsDialog;
// @ts-ignore
const __VLS_314 = __VLS_asFunctionalComponent1(__VLS_313, new __VLS_313({
    modelValue: (__VLS_ctx.logisticsDialogVisible),
}));
const __VLS_315 = __VLS_314({
    modelValue: (__VLS_ctx.logisticsDialogVisible),
}, ...__VLS_functionalComponentArgsRest(__VLS_314));
// @ts-ignore
var __VLS_148 = __VLS_147;
// @ts-ignore
[logisticsDialogVisible,];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
