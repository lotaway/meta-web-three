/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { Warning } from '@element-plus/icons-vue';
import { getOrderDetailByIdAPI, orderUpdateReceiverInfoAPI, orderUpdateMoneyInfoAPI, orderUpdateCloseAPI, orderUpdateNoteAPI, orderDeleteByIdsAPI } from '@/apis/order';
import LogisticsDialog from '@/views/oms/order/components/logisticsDialog.vue';
import { formatDateTime } from '@/utils/datetime';
import { pcaTextArr } from 'element-china-area-data';
// 获取路由对象
const router = useRouter();
const route = useRoute();
// 订单ID
const id = ref();
// 订单详情数据
const order = ref({});
// 组件挂载后获取订单详情
onMounted(async () => {
    id.value = Number(route.query.id);
    const res = await getOrderDetailByIdAPI(id.value);
    order.value = res.data;
});
// 修改收货人信息对话框可见性
const receiverDialogVisible = ref(false);
// 收货人信息
const receiverInfo = ref({});
// 当前选中的省市区三级联动
const selectedRegions = ref([]);
// 修改费用信息对话框可见性
const moneyDialogVisible = ref(false);
// 费用信息
const moneyInfo = ref({ orderId: 0, freightAmount: 0, discountAmount: 0, status: 0 });
// 发送站内信对话框可见性
const messageDialogVisible = ref(false);
// 站内信内容
const message = ref({ title: '', content: '' });
// 关闭订单对话框可见性
const closeDialogVisible = ref(false);
// 关闭订单信息
const closeInfo = ref({ note: '', id: 0 });
// 备注订单对话框可见性
const markOrderDialogVisible = ref(false);
// 备注订单信息
const markInfo = ref({ id: 0, note: '' });
// 物流对话框可见性
const logisticsDialogVisible = ref(false);
// 格式化空值
const formatNull = (value) => {
    if (!value) {
        return '暂无';
    }
    else {
        return value;
    }
};
// 格式化长文本
const formatLongText = (value) => {
    if (!value) {
        return '暂无';
    }
    else if (value.length > 8) {
        return value.substring(0, 8) + '...';
    }
    else {
        return value;
    }
};
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
// 格式化订单类型
const formatOrderType = (value) => {
    if (value === 1) {
        return '秒杀订单';
    }
    else {
        return '正常订单';
    }
};
// 格式化地址
const formatAddress = (order) => {
    let str = order.receiverProvince || '';
    if (order.receiverCity != null) {
        str += "  " + order.receiverCity;
    }
    str += "  " + (order.receiverRegion || '');
    str += "  " + (order.receiverDetailAddress || '');
    return str;
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
// 格式化支付状态
const formatPayStatus = (value) => {
    if (value === 0) {
        return '未支付';
    }
    else if (value === 4) {
        return '已退款';
    }
    else {
        return '已支付';
    }
};
// 格式化发货状态
const formatDeliverStatus = (value) => {
    if (value === 0 || value === 1) {
        return '未发货';
    }
    else {
        return '已发货';
    }
};
// 格式化商品属性
const formatProductAttr = (value) => {
    if (value == null) {
        return '';
    }
    else {
        const attr = JSON.parse(value);
        let result = '';
        for (let i = 0; i < attr.length; i++) {
            result += attr[i].key;
            result += ":";
            result += attr[i].value;
            result += ";";
        }
        return result;
    }
};
// 格式化步骤状态
const formatStepStatus = (value) => {
    if (value === 1) {
        //待发货
        return 2;
    }
    else if (value === 2) {
        //已发货
        return 3;
    }
    else if (value === 3) {
        //已完成
        return 4;
    }
    else {
        //待付款、已关闭、无限订单
        return 1;
    }
};
// 选择地区
const onSelectRegionChange = () => {
    receiverInfo.value.receiverProvince = selectedRegions.value[0];
    receiverInfo.value.receiverCity = selectedRegions.value[1];
    receiverInfo.value.receiverRegion = selectedRegions.value[2];
};
// 显示修改收货人信息对话框
const showUpdateReceiverDialog = () => {
    receiverDialogVisible.value = true;
    receiverInfo.value = {
        orderId: order.value.id,
        receiverName: order.value.receiverName,
        receiverPhone: order.value.receiverPhone,
        receiverPostCode: order.value.receiverPostCode,
        receiverDetailAddress: order.value.receiverDetailAddress,
        receiverProvince: order.value.receiverProvince,
        receiverCity: order.value.receiverCity,
        receiverRegion: order.value.receiverRegion,
        status: order.value.status
    };
    // 初始化地址选择器中数据
    selectedRegions.value = [];
    selectedRegions.value[0] = receiverInfo.value.receiverProvince;
    selectedRegions.value[1] = receiverInfo.value.receiverCity;
    selectedRegions.value[2] = receiverInfo.value.receiverRegion;
};
// 处理更新收货人信息
const handleUpdateReceiverInfo = async () => {
    await ElMessageBox.confirm('是否要修改收货信息?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    await orderUpdateReceiverInfoAPI(receiverInfo.value);
    receiverDialogVisible.value = false;
    ElMessage({
        type: 'success',
        message: '修改成功!'
    });
    const response = await getOrderDetailByIdAPI(id.value);
    order.value = response.data;
};
// 显示修改费用信息对话框
const showUpdateMoneyDialog = () => {
    moneyDialogVisible.value = true;
    moneyInfo.value.orderId = order.value.id;
    moneyInfo.value.freightAmount = order.value.freightAmount;
    moneyInfo.value.discountAmount = order.value.discountAmount;
    moneyInfo.value.status = order.value.status;
};
// 处理更新费用信息
const handleUpdateMoneyInfo = async () => {
    await ElMessageBox.confirm('是否要修改费用信息?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    await orderUpdateMoneyInfoAPI(moneyInfo.value);
    moneyDialogVisible.value = false;
    ElMessage({
        type: 'success',
        message: '修改成功!'
    });
    const response = await getOrderDetailByIdAPI(id.value);
    order.value = response.data;
};
// 显示发送站内信对话框
const showMessageDialog = () => {
    messageDialogVisible.value = true;
    message.value.title = '';
    message.value.content = '';
};
// 处理发送站内信
const handleSendMessage = async () => {
    await ElMessageBox.confirm('是否要发送站内信?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    console.log('站内信功能暂未实现，模拟发送。。。');
    messageDialogVisible.value = false;
    ElMessage({
        type: 'success',
        message: '发送成功!'
    });
};
// 显示关闭订单对话框
const showCloseOrderDialog = () => {
    closeDialogVisible.value = true;
    closeInfo.value.note = '';
    closeInfo.value.id = id.value;
};
// 处理关闭订单
const handleCloseOrder = async () => {
    try {
        await ElMessageBox.confirm('是否要关闭?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await orderUpdateCloseAPI({ ids: closeInfo.value.id.toString(), note: closeInfo.value.note });
        closeDialogVisible.value = false;
        ElMessage({
            type: 'success',
            message: '订单关闭成功!'
        });
        const response = await getOrderDetailByIdAPI(id.value);
        order.value = response.data;
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('关闭订单失败:', error);
        }
    }
};
// 显示备注订单对话框
const showMarkOrderDialog = () => {
    markOrderDialogVisible.value = true;
    markInfo.value.id = id.value;
    markInfo.value.note = '';
};
// 处理备注订单
const handleMarkOrder = async () => {
    await ElMessageBox.confirm('是否要备注订单?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    await orderUpdateNoteAPI({ id: markInfo.value.id, note: markInfo.value.note, status: order.value.status });
    markOrderDialogVisible.value = false;
    ElMessage({
        type: 'success',
        message: '订单备注成功!'
    });
    const response = await getOrderDetailByIdAPI(id.value);
    order.value = response.data;
};
// 处理删除订单
const handleDeleteOrder = async () => {
    try {
        await ElMessageBox.confirm('是否要进行该删除操作?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await orderDeleteByIdsAPI({ ids: id.value.toString() });
        ElMessage({
            message: '删除成功！',
            type: 'success',
            duration: 1000
        });
        router.back();
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('删除订单失败:', error);
        }
    }
};
// 显示物流对话框
const showLogisticsDialog = () => {
    logisticsDialogVisible.value = true;
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
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elSteps | typeof __VLS_components.ElSteps | typeof __VLS_components['el-steps'] | typeof __VLS_components.elSteps | typeof __VLS_components.ElSteps | typeof __VLS_components['el-steps']} */
elSteps;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    active: (__VLS_ctx.formatStepStatus(__VLS_ctx.order.status)),
    finishStatus: "success",
    alignCenter: true,
}));
const __VLS_2 = __VLS_1({
    active: (__VLS_ctx.formatStepStatus(__VLS_ctx.order.status)),
    finishStatus: "success",
    alignCenter: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
const { default: __VLS_5 } = __VLS_3.slots;
let __VLS_6;
/** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
elStep;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({
    title: "提交订单",
    description: (__VLS_ctx.order.createTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.createTime) : ''),
}));
const __VLS_8 = __VLS_7({
    title: "提交订单",
    description: (__VLS_ctx.order.createTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.createTime) : ''),
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
let __VLS_11;
/** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
elStep;
// @ts-ignore
const __VLS_12 = __VLS_asFunctionalComponent1(__VLS_11, new __VLS_11({
    title: "支付订单",
    description: (__VLS_ctx.order.paymentTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.paymentTime) : ''),
}));
const __VLS_13 = __VLS_12({
    title: "支付订单",
    description: (__VLS_ctx.order.paymentTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.paymentTime) : ''),
}, ...__VLS_functionalComponentArgsRest(__VLS_12));
let __VLS_16;
/** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
elStep;
// @ts-ignore
const __VLS_17 = __VLS_asFunctionalComponent1(__VLS_16, new __VLS_16({
    title: "平台发货",
    description: (__VLS_ctx.order.deliveryTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.deliveryTime) : ''),
}));
const __VLS_18 = __VLS_17({
    title: "平台发货",
    description: (__VLS_ctx.order.deliveryTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.deliveryTime) : ''),
}, ...__VLS_functionalComponentArgsRest(__VLS_17));
let __VLS_21;
/** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
elStep;
// @ts-ignore
const __VLS_22 = __VLS_asFunctionalComponent1(__VLS_21, new __VLS_21({
    title: "确认收货",
    description: (__VLS_ctx.order.receiveTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.receiveTime) : ''),
}));
const __VLS_23 = __VLS_22({
    title: "确认收货",
    description: (__VLS_ctx.order.receiveTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.receiveTime) : ''),
}, ...__VLS_functionalComponentArgsRest(__VLS_22));
let __VLS_26;
/** @ts-ignore @type { | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step'] | typeof __VLS_components.elStep | typeof __VLS_components.ElStep | typeof __VLS_components['el-step']} */
elStep;
// @ts-ignore
const __VLS_27 = __VLS_asFunctionalComponent1(__VLS_26, new __VLS_26({
    title: "完成评价",
    description: (__VLS_ctx.order.commentTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.commentTime) : ''),
}));
const __VLS_28 = __VLS_27({
    title: "完成评价",
    description: (__VLS_ctx.order.commentTime ? __VLS_ctx.formatDateTime(__VLS_ctx.order.commentTime) : ''),
}, ...__VLS_functionalComponentArgsRest(__VLS_27));
// @ts-ignore
[formatStepStatus, order, order, order, order, order, order, order, order, order, order, order, formatDateTime, formatDateTime, formatDateTime, formatDateTime, formatDateTime,];
var __VLS_3;
let __VLS_31;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_32 = __VLS_asFunctionalComponent1(__VLS_31, new __VLS_31({
    shadow: "never",
    ...{ style: {} },
}));
const __VLS_33 = __VLS_32({
    shadow: "never",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_32));
const { default: __VLS_36 } = __VLS_34.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "operate-container" },
});
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
let __VLS_37;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_38 = __VLS_asFunctionalComponent1(__VLS_37, new __VLS_37({
    ...{ class: "color-danger el-icon-middle" },
    ...{ style: {} },
}));
const __VLS_39 = __VLS_38({
    ...{ class: "color-danger el-icon-middle" },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_38));
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_42 } = __VLS_40.slots;
let __VLS_43;
/** @ts-ignore @type { | typeof __VLS_components.Warning} */
Warning;
// @ts-ignore
const __VLS_44 = __VLS_asFunctionalComponent1(__VLS_43, new __VLS_43({}));
const __VLS_45 = __VLS_44({}, ...__VLS_functionalComponentArgsRest(__VLS_44));
// @ts-ignore
[];
var __VLS_40;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
(__VLS_ctx.formatStatus(__VLS_ctx.order.status));
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "operate-button-container" },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.order.status === 0) }, null, null);
/** @type {__VLS_StyleScopedClasses['operate-button-container']} */ ;
let __VLS_48;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_49 = __VLS_asFunctionalComponent1(__VLS_48, new __VLS_48({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_50 = __VLS_49({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_49));
let __VLS_53;
const __VLS_54 = ({ click: {} },
    { onClick: (__VLS_ctx.showUpdateReceiverDialog) });
const { default: __VLS_55 } = __VLS_51.slots;
// @ts-ignore
[order, order, formatStatus, showUpdateReceiverDialog,];
var __VLS_51;
var __VLS_52;
let __VLS_56;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_58 = __VLS_57({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_57));
let __VLS_61;
const __VLS_62 = ({ click: {} },
    { onClick: (__VLS_ctx.showUpdateMoneyDialog) });
const { default: __VLS_63 } = __VLS_59.slots;
// @ts-ignore
[showUpdateMoneyDialog,];
var __VLS_59;
var __VLS_60;
let __VLS_64;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_65 = __VLS_asFunctionalComponent1(__VLS_64, new __VLS_64({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_66 = __VLS_65({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_65));
let __VLS_69;
const __VLS_70 = ({ click: {} },
    { onClick: (__VLS_ctx.showMessageDialog) });
const { default: __VLS_71 } = __VLS_67.slots;
// @ts-ignore
[showMessageDialog,];
var __VLS_67;
var __VLS_68;
let __VLS_72;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_73 = __VLS_asFunctionalComponent1(__VLS_72, new __VLS_72({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_74 = __VLS_73({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_73));
let __VLS_77;
const __VLS_78 = ({ click: {} },
    { onClick: (__VLS_ctx.showCloseOrderDialog) });
const { default: __VLS_79 } = __VLS_75.slots;
// @ts-ignore
[showCloseOrderDialog,];
var __VLS_75;
var __VLS_76;
let __VLS_80;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_81 = __VLS_asFunctionalComponent1(__VLS_80, new __VLS_80({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_82 = __VLS_81({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_81));
let __VLS_85;
const __VLS_86 = ({ click: {} },
    { onClick: (__VLS_ctx.showMarkOrderDialog) });
const { default: __VLS_87 } = __VLS_83.slots;
// @ts-ignore
[showMarkOrderDialog,];
var __VLS_83;
var __VLS_84;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "operate-button-container" },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.order.status === 1) }, null, null);
/** @type {__VLS_StyleScopedClasses['operate-button-container']} */ ;
let __VLS_88;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_89 = __VLS_asFunctionalComponent1(__VLS_88, new __VLS_88({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_90 = __VLS_89({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_89));
let __VLS_93;
const __VLS_94 = ({ click: {} },
    { onClick: (__VLS_ctx.showUpdateReceiverDialog) });
const { default: __VLS_95 } = __VLS_91.slots;
// @ts-ignore
[order, showUpdateReceiverDialog,];
var __VLS_91;
var __VLS_92;
let __VLS_96;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_97 = __VLS_asFunctionalComponent1(__VLS_96, new __VLS_96({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_98 = __VLS_97({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_97));
let __VLS_101;
const __VLS_102 = ({ click: {} },
    { onClick: (__VLS_ctx.showMessageDialog) });
const { default: __VLS_103 } = __VLS_99.slots;
// @ts-ignore
[showMessageDialog,];
var __VLS_99;
var __VLS_100;
let __VLS_104;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_105 = __VLS_asFunctionalComponent1(__VLS_104, new __VLS_104({
    size: "small",
}));
const __VLS_106 = __VLS_105({
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_105));
const { default: __VLS_109 } = __VLS_107.slots;
// @ts-ignore
[];
var __VLS_107;
let __VLS_110;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_111 = __VLS_asFunctionalComponent1(__VLS_110, new __VLS_110({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_112 = __VLS_111({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_111));
let __VLS_115;
const __VLS_116 = ({ click: {} },
    { onClick: (__VLS_ctx.showMarkOrderDialog) });
const { default: __VLS_117 } = __VLS_113.slots;
// @ts-ignore
[showMarkOrderDialog,];
var __VLS_113;
var __VLS_114;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "operate-button-container" },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.order.status === 2 || __VLS_ctx.order.status === 3) }, null, null);
/** @type {__VLS_StyleScopedClasses['operate-button-container']} */ ;
let __VLS_118;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_119 = __VLS_asFunctionalComponent1(__VLS_118, new __VLS_118({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_120 = __VLS_119({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_119));
let __VLS_123;
const __VLS_124 = ({ click: {} },
    { onClick: (__VLS_ctx.showLogisticsDialog) });
const { default: __VLS_125 } = __VLS_121.slots;
// @ts-ignore
[order, order, showLogisticsDialog,];
var __VLS_121;
var __VLS_122;
let __VLS_126;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_127 = __VLS_asFunctionalComponent1(__VLS_126, new __VLS_126({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_128 = __VLS_127({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_127));
let __VLS_131;
const __VLS_132 = ({ click: {} },
    { onClick: (__VLS_ctx.showMessageDialog) });
const { default: __VLS_133 } = __VLS_129.slots;
// @ts-ignore
[showMessageDialog,];
var __VLS_129;
var __VLS_130;
let __VLS_134;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_135 = __VLS_asFunctionalComponent1(__VLS_134, new __VLS_134({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_136 = __VLS_135({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_135));
let __VLS_139;
const __VLS_140 = ({ click: {} },
    { onClick: (__VLS_ctx.showMarkOrderDialog) });
const { default: __VLS_141 } = __VLS_137.slots;
// @ts-ignore
[showMarkOrderDialog,];
var __VLS_137;
var __VLS_138;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "operate-button-container" },
});
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.order.status === 4) }, null, null);
/** @type {__VLS_StyleScopedClasses['operate-button-container']} */ ;
let __VLS_142;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_143 = __VLS_asFunctionalComponent1(__VLS_142, new __VLS_142({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_144 = __VLS_143({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_143));
let __VLS_147;
const __VLS_148 = ({ click: {} },
    { onClick: (__VLS_ctx.handleDeleteOrder) });
const { default: __VLS_149 } = __VLS_145.slots;
// @ts-ignore
[order, handleDeleteOrder,];
var __VLS_145;
var __VLS_146;
let __VLS_150;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_151 = __VLS_asFunctionalComponent1(__VLS_150, new __VLS_150({
    ...{ 'onClick': {} },
    size: "small",
}));
const __VLS_152 = __VLS_151({
    ...{ 'onClick': {} },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_151));
let __VLS_155;
const __VLS_156 = ({ click: {} },
    { onClick: (__VLS_ctx.showMarkOrderDialog) });
const { default: __VLS_157 } = __VLS_153.slots;
// @ts-ignore
[showMarkOrderDialog,];
var __VLS_153;
var __VLS_154;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_158;
/** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
svgIcon;
// @ts-ignore
const __VLS_159 = __VLS_asFunctionalComponent1(__VLS_158, new __VLS_158({
    iconClass: "marker",
    ...{ style: {} },
}));
const __VLS_160 = __VLS_159({
    iconClass: "marker",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_159));
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-small" },
});
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-layout" },
});
/** @type {__VLS_StyleScopedClasses['table-layout']} */ ;
let __VLS_163;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_164 = __VLS_asFunctionalComponent1(__VLS_163, new __VLS_163({}));
const __VLS_165 = __VLS_164({}, ...__VLS_functionalComponentArgsRest(__VLS_164));
const { default: __VLS_168 } = __VLS_166.slots;
let __VLS_169;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_170 = __VLS_asFunctionalComponent1(__VLS_169, new __VLS_169({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_171 = __VLS_170({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_170));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_174 } = __VLS_172.slots;
// @ts-ignore
[];
var __VLS_172;
let __VLS_175;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_176 = __VLS_asFunctionalComponent1(__VLS_175, new __VLS_175({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_177 = __VLS_176({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_176));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_180 } = __VLS_178.slots;
// @ts-ignore
[];
var __VLS_178;
let __VLS_181;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_182 = __VLS_asFunctionalComponent1(__VLS_181, new __VLS_181({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_183 = __VLS_182({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_182));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_186 } = __VLS_184.slots;
// @ts-ignore
[];
var __VLS_184;
let __VLS_187;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_188 = __VLS_asFunctionalComponent1(__VLS_187, new __VLS_187({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_189 = __VLS_188({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_188));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_192 } = __VLS_190.slots;
// @ts-ignore
[];
var __VLS_190;
let __VLS_193;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_194 = __VLS_asFunctionalComponent1(__VLS_193, new __VLS_193({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_195 = __VLS_194({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_194));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_198 } = __VLS_196.slots;
// @ts-ignore
[];
var __VLS_196;
let __VLS_199;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_200 = __VLS_asFunctionalComponent1(__VLS_199, new __VLS_199({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_201 = __VLS_200({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_200));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_204 } = __VLS_202.slots;
// @ts-ignore
[];
var __VLS_202;
// @ts-ignore
[];
var __VLS_166;
let __VLS_205;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_206 = __VLS_asFunctionalComponent1(__VLS_205, new __VLS_205({}));
const __VLS_207 = __VLS_206({}, ...__VLS_functionalComponentArgsRest(__VLS_206));
const { default: __VLS_210 } = __VLS_208.slots;
let __VLS_211;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_212 = __VLS_asFunctionalComponent1(__VLS_211, new __VLS_211({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_213 = __VLS_212({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_212));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_216 } = __VLS_214.slots;
(__VLS_ctx.order.orderSn);
// @ts-ignore
[order,];
var __VLS_214;
let __VLS_217;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_218 = __VLS_asFunctionalComponent1(__VLS_217, new __VLS_217({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_219 = __VLS_218({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_218));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_222 } = __VLS_220.slots;
// @ts-ignore
[];
var __VLS_220;
let __VLS_223;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_224 = __VLS_asFunctionalComponent1(__VLS_223, new __VLS_223({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_225 = __VLS_224({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_224));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_228 } = __VLS_226.slots;
(__VLS_ctx.order.memberUsername);
// @ts-ignore
[order,];
var __VLS_226;
let __VLS_229;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_230 = __VLS_asFunctionalComponent1(__VLS_229, new __VLS_229({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_231 = __VLS_230({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_230));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_234 } = __VLS_232.slots;
(__VLS_ctx.formatPayType(__VLS_ctx.order.payType));
// @ts-ignore
[order, formatPayType,];
var __VLS_232;
let __VLS_235;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_236 = __VLS_asFunctionalComponent1(__VLS_235, new __VLS_235({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_237 = __VLS_236({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_236));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_240 } = __VLS_238.slots;
(__VLS_ctx.formatSourceType(__VLS_ctx.order.sourceType));
// @ts-ignore
[order, formatSourceType,];
var __VLS_238;
let __VLS_241;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_242 = __VLS_asFunctionalComponent1(__VLS_241, new __VLS_241({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_243 = __VLS_242({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_242));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_246 } = __VLS_244.slots;
(__VLS_ctx.formatOrderType(__VLS_ctx.order.orderType));
// @ts-ignore
[order, formatOrderType,];
var __VLS_244;
// @ts-ignore
[];
var __VLS_208;
let __VLS_247;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_248 = __VLS_asFunctionalComponent1(__VLS_247, new __VLS_247({}));
const __VLS_249 = __VLS_248({}, ...__VLS_functionalComponentArgsRest(__VLS_248));
const { default: __VLS_252 } = __VLS_250.slots;
let __VLS_253;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_254 = __VLS_asFunctionalComponent1(__VLS_253, new __VLS_253({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_255 = __VLS_254({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_254));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_258 } = __VLS_256.slots;
// @ts-ignore
[];
var __VLS_256;
let __VLS_259;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_260 = __VLS_asFunctionalComponent1(__VLS_259, new __VLS_259({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_261 = __VLS_260({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_260));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_264 } = __VLS_262.slots;
// @ts-ignore
[];
var __VLS_262;
let __VLS_265;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_266 = __VLS_asFunctionalComponent1(__VLS_265, new __VLS_265({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_267 = __VLS_266({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_266));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_270 } = __VLS_268.slots;
// @ts-ignore
[];
var __VLS_268;
let __VLS_271;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_272 = __VLS_asFunctionalComponent1(__VLS_271, new __VLS_271({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_273 = __VLS_272({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_272));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_276 } = __VLS_274.slots;
// @ts-ignore
[];
var __VLS_274;
let __VLS_277;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_278 = __VLS_asFunctionalComponent1(__VLS_277, new __VLS_277({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_279 = __VLS_278({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_278));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_282 } = __VLS_280.slots;
// @ts-ignore
[];
var __VLS_280;
let __VLS_283;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_284 = __VLS_asFunctionalComponent1(__VLS_283, new __VLS_283({
    span: (4),
    ...{ class: "table-cell-title" },
}));
const __VLS_285 = __VLS_284({
    span: (4),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_284));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_288 } = __VLS_286.slots;
// @ts-ignore
[];
var __VLS_286;
// @ts-ignore
[];
var __VLS_250;
let __VLS_289;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_290 = __VLS_asFunctionalComponent1(__VLS_289, new __VLS_289({}));
const __VLS_291 = __VLS_290({}, ...__VLS_functionalComponentArgsRest(__VLS_290));
const { default: __VLS_294 } = __VLS_292.slots;
let __VLS_295;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_296 = __VLS_asFunctionalComponent1(__VLS_295, new __VLS_295({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_297 = __VLS_296({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_296));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_300 } = __VLS_298.slots;
(__VLS_ctx.formatNull(__VLS_ctx.order.deliveryCompany));
// @ts-ignore
[order, formatNull,];
var __VLS_298;
let __VLS_301;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_302 = __VLS_asFunctionalComponent1(__VLS_301, new __VLS_301({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_303 = __VLS_302({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_302));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_306 } = __VLS_304.slots;
(__VLS_ctx.formatNull(__VLS_ctx.order.deliverySn));
// @ts-ignore
[order, formatNull,];
var __VLS_304;
let __VLS_307;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_308 = __VLS_asFunctionalComponent1(__VLS_307, new __VLS_307({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_309 = __VLS_308({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_308));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_312 } = __VLS_310.slots;
(__VLS_ctx.order.autoConfirmDay);
// @ts-ignore
[order,];
var __VLS_310;
let __VLS_313;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_314 = __VLS_asFunctionalComponent1(__VLS_313, new __VLS_313({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_315 = __VLS_314({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_314));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_318 } = __VLS_316.slots;
(__VLS_ctx.order.integration);
// @ts-ignore
[order,];
var __VLS_316;
let __VLS_319;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_320 = __VLS_asFunctionalComponent1(__VLS_319, new __VLS_319({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_321 = __VLS_320({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_320));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_324 } = __VLS_322.slots;
(__VLS_ctx.order.growth);
// @ts-ignore
[order,];
var __VLS_322;
let __VLS_325;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_326 = __VLS_asFunctionalComponent1(__VLS_325, new __VLS_325({
    span: (4),
    ...{ class: "table-cell" },
}));
const __VLS_327 = __VLS_326({
    span: (4),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_326));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_330 } = __VLS_328.slots;
let __VLS_331;
/** @ts-ignore @type { | typeof __VLS_components.elPopover | typeof __VLS_components.ElPopover | typeof __VLS_components['el-popover'] | typeof __VLS_components.elPopover | typeof __VLS_components.ElPopover | typeof __VLS_components['el-popover']} */
elPopover;
// @ts-ignore
const __VLS_332 = __VLS_asFunctionalComponent1(__VLS_331, new __VLS_331({
    placement: "top-start",
    title: "活动信息",
    width: "200",
    trigger: "hover",
    content: (__VLS_ctx.order.promotionInfo),
}));
const __VLS_333 = __VLS_332({
    placement: "top-start",
    title: "活动信息",
    width: "200",
    trigger: "hover",
    content: (__VLS_ctx.order.promotionInfo),
}, ...__VLS_functionalComponentArgsRest(__VLS_332));
const { default: __VLS_336 } = __VLS_334.slots;
{
    const { reference: __VLS_337 } = __VLS_334.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
    (__VLS_ctx.formatLongText(__VLS_ctx.order.promotionInfo));
    // @ts-ignore
    [order, order, formatLongText,];
}
// @ts-ignore
[];
var __VLS_334;
// @ts-ignore
[];
var __VLS_328;
// @ts-ignore
[];
var __VLS_292;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_338;
/** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
svgIcon;
// @ts-ignore
const __VLS_339 = __VLS_asFunctionalComponent1(__VLS_338, new __VLS_338({
    iconClass: "marker",
    ...{ style: {} },
}));
const __VLS_340 = __VLS_339({
    iconClass: "marker",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_339));
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-small" },
});
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-layout" },
});
/** @type {__VLS_StyleScopedClasses['table-layout']} */ ;
let __VLS_343;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_344 = __VLS_asFunctionalComponent1(__VLS_343, new __VLS_343({}));
const __VLS_345 = __VLS_344({}, ...__VLS_functionalComponentArgsRest(__VLS_344));
const { default: __VLS_348 } = __VLS_346.slots;
let __VLS_349;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_350 = __VLS_asFunctionalComponent1(__VLS_349, new __VLS_349({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_351 = __VLS_350({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_350));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_354 } = __VLS_352.slots;
// @ts-ignore
[];
var __VLS_352;
let __VLS_355;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_356 = __VLS_asFunctionalComponent1(__VLS_355, new __VLS_355({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_357 = __VLS_356({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_356));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_360 } = __VLS_358.slots;
// @ts-ignore
[];
var __VLS_358;
let __VLS_361;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_362 = __VLS_asFunctionalComponent1(__VLS_361, new __VLS_361({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_363 = __VLS_362({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_362));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_366 } = __VLS_364.slots;
// @ts-ignore
[];
var __VLS_364;
let __VLS_367;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_368 = __VLS_asFunctionalComponent1(__VLS_367, new __VLS_367({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_369 = __VLS_368({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_368));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_372 } = __VLS_370.slots;
// @ts-ignore
[];
var __VLS_370;
// @ts-ignore
[];
var __VLS_346;
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
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_381 = __VLS_380({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_380));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_384 } = __VLS_382.slots;
(__VLS_ctx.order.receiverName);
// @ts-ignore
[order,];
var __VLS_382;
let __VLS_385;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_386 = __VLS_asFunctionalComponent1(__VLS_385, new __VLS_385({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_387 = __VLS_386({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_386));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_390 } = __VLS_388.slots;
(__VLS_ctx.order.receiverPhone);
// @ts-ignore
[order,];
var __VLS_388;
let __VLS_391;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_392 = __VLS_asFunctionalComponent1(__VLS_391, new __VLS_391({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_393 = __VLS_392({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_392));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_396 } = __VLS_394.slots;
(__VLS_ctx.order.receiverPostCode);
// @ts-ignore
[order,];
var __VLS_394;
let __VLS_397;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_398 = __VLS_asFunctionalComponent1(__VLS_397, new __VLS_397({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_399 = __VLS_398({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_398));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_402 } = __VLS_400.slots;
(__VLS_ctx.formatAddress(__VLS_ctx.order));
// @ts-ignore
[order, formatAddress,];
var __VLS_400;
// @ts-ignore
[];
var __VLS_376;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_403;
/** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
svgIcon;
// @ts-ignore
const __VLS_404 = __VLS_asFunctionalComponent1(__VLS_403, new __VLS_403({
    iconClass: "marker",
    ...{ style: {} },
}));
const __VLS_405 = __VLS_404({
    iconClass: "marker",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_404));
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-small" },
});
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
let __VLS_408;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_409 = __VLS_asFunctionalComponent1(__VLS_408, new __VLS_408({
    ref: "orderItemTable",
    data: (__VLS_ctx.order.orderItemList),
    ...{ style: {} },
    border: true,
}));
const __VLS_410 = __VLS_409({
    ref: "orderItemTable",
    data: (__VLS_ctx.order.orderItemList),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_409));
var __VLS_413 = {};
const { default: __VLS_415 } = __VLS_411.slots;
let __VLS_416;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_417 = __VLS_asFunctionalComponent1(__VLS_416, new __VLS_416({
    label: "商品图片",
    width: "120",
    align: "center",
}));
const __VLS_418 = __VLS_417({
    label: "商品图片",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_417));
const { default: __VLS_421 } = __VLS_419.slots;
{
    const { default: __VLS_422 } = __VLS_419.slots;
    const [scope] = __VLS_vSlot(__VLS_422);
    __VLS_asFunctionalElement1(__VLS_intrinsics.img)({
        src: (scope.row.productPic),
        ...{ style: {} },
    });
    // @ts-ignore
    [order,];
    __VLS_419.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_419;
let __VLS_423;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_424 = __VLS_asFunctionalComponent1(__VLS_423, new __VLS_423({
    label: "商品名称",
    align: "center",
}));
const __VLS_425 = __VLS_424({
    label: "商品名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_424));
const { default: __VLS_428 } = __VLS_426.slots;
{
    const { default: __VLS_429 } = __VLS_426.slots;
    const [scope] = __VLS_vSlot(__VLS_429);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    (scope.row.productName);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    (scope.row.productBrand);
    // @ts-ignore
    [];
    __VLS_426.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_426;
let __VLS_430;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_431 = __VLS_asFunctionalComponent1(__VLS_430, new __VLS_430({
    label: "价格/货号",
    width: "160",
    align: "center",
}));
const __VLS_432 = __VLS_431({
    label: "价格/货号",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_431));
const { default: __VLS_435 } = __VLS_433.slots;
{
    const { default: __VLS_436 } = __VLS_433.slots;
    const [scope] = __VLS_vSlot(__VLS_436);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    (scope.row.productPrice);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    (scope.row.productSn);
    // @ts-ignore
    [];
    __VLS_433.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_433;
let __VLS_437;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_438 = __VLS_asFunctionalComponent1(__VLS_437, new __VLS_437({
    label: "属性",
    width: "160",
    align: "center",
}));
const __VLS_439 = __VLS_438({
    label: "属性",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_438));
const { default: __VLS_442 } = __VLS_440.slots;
{
    const { default: __VLS_443 } = __VLS_440.slots;
    const [scope] = __VLS_vSlot(__VLS_443);
    (__VLS_ctx.formatProductAttr(scope.row.productAttr));
    // @ts-ignore
    [formatProductAttr,];
    __VLS_440.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_440;
let __VLS_444;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_445 = __VLS_asFunctionalComponent1(__VLS_444, new __VLS_444({
    label: "数量",
    width: "120",
    align: "center",
}));
const __VLS_446 = __VLS_445({
    label: "数量",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_445));
const { default: __VLS_449 } = __VLS_447.slots;
{
    const { default: __VLS_450 } = __VLS_447.slots;
    const [scope] = __VLS_vSlot(__VLS_450);
    (scope.row.productQuantity);
    // @ts-ignore
    [];
    __VLS_447.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_447;
let __VLS_451;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_452 = __VLS_asFunctionalComponent1(__VLS_451, new __VLS_451({
    label: "小计",
    width: "120",
    align: "center",
}));
const __VLS_453 = __VLS_452({
    label: "小计",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_452));
const { default: __VLS_456 } = __VLS_454.slots;
{
    const { default: __VLS_457 } = __VLS_454.slots;
    const [scope] = __VLS_vSlot(__VLS_457);
    (scope.row.productPrice * scope.row.productQuantity);
    // @ts-ignore
    [];
    __VLS_454.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_454;
// @ts-ignore
[];
var __VLS_411;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
(__VLS_ctx.order.totalAmount);
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_458;
/** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
svgIcon;
// @ts-ignore
const __VLS_459 = __VLS_asFunctionalComponent1(__VLS_458, new __VLS_458({
    iconClass: "marker",
    ...{ style: {} },
}));
const __VLS_460 = __VLS_459({
    iconClass: "marker",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_459));
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-small" },
});
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-layout" },
});
/** @type {__VLS_StyleScopedClasses['table-layout']} */ ;
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
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_471 = __VLS_470({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_470));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_474 } = __VLS_472.slots;
// @ts-ignore
[order,];
var __VLS_472;
let __VLS_475;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_476 = __VLS_asFunctionalComponent1(__VLS_475, new __VLS_475({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_477 = __VLS_476({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_476));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_480 } = __VLS_478.slots;
// @ts-ignore
[];
var __VLS_478;
let __VLS_481;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_482 = __VLS_asFunctionalComponent1(__VLS_481, new __VLS_481({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_483 = __VLS_482({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_482));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_486 } = __VLS_484.slots;
// @ts-ignore
[];
var __VLS_484;
let __VLS_487;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_488 = __VLS_asFunctionalComponent1(__VLS_487, new __VLS_487({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_489 = __VLS_488({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_488));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_492 } = __VLS_490.slots;
// @ts-ignore
[];
var __VLS_490;
// @ts-ignore
[];
var __VLS_466;
let __VLS_493;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_494 = __VLS_asFunctionalComponent1(__VLS_493, new __VLS_493({}));
const __VLS_495 = __VLS_494({}, ...__VLS_functionalComponentArgsRest(__VLS_494));
const { default: __VLS_498 } = __VLS_496.slots;
let __VLS_499;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_500 = __VLS_asFunctionalComponent1(__VLS_499, new __VLS_499({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_501 = __VLS_500({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_500));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_504 } = __VLS_502.slots;
(__VLS_ctx.order.totalAmount);
// @ts-ignore
[order,];
var __VLS_502;
let __VLS_505;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_506 = __VLS_asFunctionalComponent1(__VLS_505, new __VLS_505({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_507 = __VLS_506({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_506));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_510 } = __VLS_508.slots;
(__VLS_ctx.order.freightAmount);
// @ts-ignore
[order,];
var __VLS_508;
let __VLS_511;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_512 = __VLS_asFunctionalComponent1(__VLS_511, new __VLS_511({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_513 = __VLS_512({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_512));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_516 } = __VLS_514.slots;
(__VLS_ctx.order.couponAmount);
// @ts-ignore
[order,];
var __VLS_514;
let __VLS_517;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_518 = __VLS_asFunctionalComponent1(__VLS_517, new __VLS_517({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_519 = __VLS_518({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_518));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_522 } = __VLS_520.slots;
(__VLS_ctx.order.integrationAmount);
// @ts-ignore
[order,];
var __VLS_520;
// @ts-ignore
[];
var __VLS_496;
let __VLS_523;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_524 = __VLS_asFunctionalComponent1(__VLS_523, new __VLS_523({}));
const __VLS_525 = __VLS_524({}, ...__VLS_functionalComponentArgsRest(__VLS_524));
const { default: __VLS_528 } = __VLS_526.slots;
let __VLS_529;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_530 = __VLS_asFunctionalComponent1(__VLS_529, new __VLS_529({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_531 = __VLS_530({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_530));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_534 } = __VLS_532.slots;
// @ts-ignore
[];
var __VLS_532;
let __VLS_535;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_536 = __VLS_asFunctionalComponent1(__VLS_535, new __VLS_535({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_537 = __VLS_536({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_536));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_540 } = __VLS_538.slots;
// @ts-ignore
[];
var __VLS_538;
let __VLS_541;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_542 = __VLS_asFunctionalComponent1(__VLS_541, new __VLS_541({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_543 = __VLS_542({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_542));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_546 } = __VLS_544.slots;
// @ts-ignore
[];
var __VLS_544;
let __VLS_547;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_548 = __VLS_asFunctionalComponent1(__VLS_547, new __VLS_547({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_549 = __VLS_548({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_548));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_552 } = __VLS_550.slots;
// @ts-ignore
[];
var __VLS_550;
// @ts-ignore
[];
var __VLS_526;
let __VLS_553;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_554 = __VLS_asFunctionalComponent1(__VLS_553, new __VLS_553({}));
const __VLS_555 = __VLS_554({}, ...__VLS_functionalComponentArgsRest(__VLS_554));
const { default: __VLS_558 } = __VLS_556.slots;
let __VLS_559;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_560 = __VLS_asFunctionalComponent1(__VLS_559, new __VLS_559({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_561 = __VLS_560({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_560));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_564 } = __VLS_562.slots;
(__VLS_ctx.order.promotionAmount);
// @ts-ignore
[order,];
var __VLS_562;
let __VLS_565;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_566 = __VLS_asFunctionalComponent1(__VLS_565, new __VLS_565({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_567 = __VLS_566({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_566));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_570 } = __VLS_568.slots;
(__VLS_ctx.order.discountAmount);
// @ts-ignore
[order,];
var __VLS_568;
let __VLS_571;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_572 = __VLS_asFunctionalComponent1(__VLS_571, new __VLS_571({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_573 = __VLS_572({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_572));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_576 } = __VLS_574.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
(__VLS_ctx.order.totalAmount + __VLS_ctx.order.freightAmount);
// @ts-ignore
[order, order,];
var __VLS_574;
let __VLS_577;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_578 = __VLS_asFunctionalComponent1(__VLS_577, new __VLS_577({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_579 = __VLS_578({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_578));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_582 } = __VLS_580.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
(__VLS_ctx.order.payAmount + __VLS_ctx.order.freightAmount - __VLS_ctx.order.discountAmount);
// @ts-ignore
[order, order, order,];
var __VLS_580;
// @ts-ignore
[];
var __VLS_556;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_583;
/** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
svgIcon;
// @ts-ignore
const __VLS_584 = __VLS_asFunctionalComponent1(__VLS_583, new __VLS_583({
    iconClass: "marker",
    ...{ style: {} },
}));
const __VLS_585 = __VLS_584({
    iconClass: "marker",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_584));
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-small" },
});
/** @type {__VLS_StyleScopedClasses['font-small']} */ ;
let __VLS_588;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_589 = __VLS_asFunctionalComponent1(__VLS_588, new __VLS_588({
    ...{ style: {} },
    ref: "orderHistoryTable",
    data: (__VLS_ctx.order.historyList),
    border: true,
}));
const __VLS_590 = __VLS_589({
    ...{ style: {} },
    ref: "orderHistoryTable",
    data: (__VLS_ctx.order.historyList),
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_589));
var __VLS_593 = {};
const { default: __VLS_595 } = __VLS_591.slots;
let __VLS_596;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_597 = __VLS_asFunctionalComponent1(__VLS_596, new __VLS_596({
    label: "操作者",
    width: "120",
    align: "center",
}));
const __VLS_598 = __VLS_597({
    label: "操作者",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_597));
const { default: __VLS_601 } = __VLS_599.slots;
{
    const { default: __VLS_602 } = __VLS_599.slots;
    const [scope] = __VLS_vSlot(__VLS_602);
    (scope.row.operateMan);
    // @ts-ignore
    [order,];
    __VLS_599.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_599;
let __VLS_603;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_604 = __VLS_asFunctionalComponent1(__VLS_603, new __VLS_603({
    label: "操作时间",
    width: "160",
    align: "center",
}));
const __VLS_605 = __VLS_604({
    label: "操作时间",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_604));
const { default: __VLS_608 } = __VLS_606.slots;
{
    const { default: __VLS_609 } = __VLS_606.slots;
    const [scope] = __VLS_vSlot(__VLS_609);
    (__VLS_ctx.formatDateTime(scope.row.createTime));
    // @ts-ignore
    [formatDateTime,];
    __VLS_606.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_606;
let __VLS_610;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_611 = __VLS_asFunctionalComponent1(__VLS_610, new __VLS_610({
    label: "订单状态",
    width: "120",
    align: "center",
}));
const __VLS_612 = __VLS_611({
    label: "订单状态",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_611));
const { default: __VLS_615 } = __VLS_613.slots;
{
    const { default: __VLS_616 } = __VLS_613.slots;
    const [scope] = __VLS_vSlot(__VLS_616);
    (__VLS_ctx.formatStatus(scope.row.orderStatus));
    // @ts-ignore
    [formatStatus,];
    __VLS_613.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_613;
let __VLS_617;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_618 = __VLS_asFunctionalComponent1(__VLS_617, new __VLS_617({
    label: "付款状态",
    width: "120",
    align: "center",
}));
const __VLS_619 = __VLS_618({
    label: "付款状态",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_618));
const { default: __VLS_622 } = __VLS_620.slots;
{
    const { default: __VLS_623 } = __VLS_620.slots;
    const [scope] = __VLS_vSlot(__VLS_623);
    (__VLS_ctx.formatPayStatus(scope.row.orderStatus));
    // @ts-ignore
    [formatPayStatus,];
    __VLS_620.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_620;
let __VLS_624;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_625 = __VLS_asFunctionalComponent1(__VLS_624, new __VLS_624({
    label: "发货状态",
    width: "120",
    align: "center",
}));
const __VLS_626 = __VLS_625({
    label: "发货状态",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_625));
const { default: __VLS_629 } = __VLS_627.slots;
{
    const { default: __VLS_630 } = __VLS_627.slots;
    const [scope] = __VLS_vSlot(__VLS_630);
    (__VLS_ctx.formatDeliverStatus(scope.row.orderStatus));
    // @ts-ignore
    [formatDeliverStatus,];
    __VLS_627.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_627;
let __VLS_631;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_632 = __VLS_asFunctionalComponent1(__VLS_631, new __VLS_631({
    label: "备注",
    align: "center",
}));
const __VLS_633 = __VLS_632({
    label: "备注",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_632));
const { default: __VLS_636 } = __VLS_634.slots;
{
    const { default: __VLS_637 } = __VLS_634.slots;
    const [scope] = __VLS_vSlot(__VLS_637);
    (scope.row.note);
    // @ts-ignore
    [];
    __VLS_634.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_634;
// @ts-ignore
[];
var __VLS_591;
// @ts-ignore
[];
var __VLS_34;
let __VLS_638;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_639 = __VLS_asFunctionalComponent1(__VLS_638, new __VLS_638({
    title: "修改收货人信息",
    modelValue: (__VLS_ctx.receiverDialogVisible),
    width: "40%",
}));
const __VLS_640 = __VLS_639({
    title: "修改收货人信息",
    modelValue: (__VLS_ctx.receiverDialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_639));
const { default: __VLS_643 } = __VLS_641.slots;
let __VLS_644;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_645 = __VLS_asFunctionalComponent1(__VLS_644, new __VLS_644({
    model: (__VLS_ctx.receiverInfo),
    ref: "receiverInfoForm",
    labelWidth: "150px",
}));
const __VLS_646 = __VLS_645({
    model: (__VLS_ctx.receiverInfo),
    ref: "receiverInfoForm",
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_645));
var __VLS_649 = {};
const { default: __VLS_651 } = __VLS_647.slots;
let __VLS_652;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_653 = __VLS_asFunctionalComponent1(__VLS_652, new __VLS_652({
    label: "收货人姓名：",
}));
const __VLS_654 = __VLS_653({
    label: "收货人姓名：",
}, ...__VLS_functionalComponentArgsRest(__VLS_653));
const { default: __VLS_657 } = __VLS_655.slots;
let __VLS_658;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_659 = __VLS_asFunctionalComponent1(__VLS_658, new __VLS_658({
    modelValue: (__VLS_ctx.receiverInfo.receiverName),
    ...{ style: {} },
}));
const __VLS_660 = __VLS_659({
    modelValue: (__VLS_ctx.receiverInfo.receiverName),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_659));
// @ts-ignore
[receiverDialogVisible, receiverInfo, receiverInfo,];
var __VLS_655;
let __VLS_663;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_664 = __VLS_asFunctionalComponent1(__VLS_663, new __VLS_663({
    label: "手机号码：",
}));
const __VLS_665 = __VLS_664({
    label: "手机号码：",
}, ...__VLS_functionalComponentArgsRest(__VLS_664));
const { default: __VLS_668 } = __VLS_666.slots;
let __VLS_669;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_670 = __VLS_asFunctionalComponent1(__VLS_669, new __VLS_669({
    modelValue: (__VLS_ctx.receiverInfo.receiverPhone),
    ...{ style: {} },
}));
const __VLS_671 = __VLS_670({
    modelValue: (__VLS_ctx.receiverInfo.receiverPhone),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_670));
// @ts-ignore
[receiverInfo,];
var __VLS_666;
let __VLS_674;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_675 = __VLS_asFunctionalComponent1(__VLS_674, new __VLS_674({
    label: "邮政编码：",
}));
const __VLS_676 = __VLS_675({
    label: "邮政编码：",
}, ...__VLS_functionalComponentArgsRest(__VLS_675));
const { default: __VLS_679 } = __VLS_677.slots;
let __VLS_680;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_681 = __VLS_asFunctionalComponent1(__VLS_680, new __VLS_680({
    modelValue: (__VLS_ctx.receiverInfo.receiverPostCode),
    ...{ style: {} },
}));
const __VLS_682 = __VLS_681({
    modelValue: (__VLS_ctx.receiverInfo.receiverPostCode),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_681));
// @ts-ignore
[receiverInfo,];
var __VLS_677;
let __VLS_685;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_686 = __VLS_asFunctionalComponent1(__VLS_685, new __VLS_685({
    label: "所在区域：",
}));
const __VLS_687 = __VLS_686({
    label: "所在区域：",
}, ...__VLS_functionalComponentArgsRest(__VLS_686));
const { default: __VLS_690 } = __VLS_688.slots;
let __VLS_691;
/** @ts-ignore @type { | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader'] | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader']} */
elCascader;
// @ts-ignore
const __VLS_692 = __VLS_asFunctionalComponent1(__VLS_691, new __VLS_691({
    ...{ 'onChange': {} },
    modelValue: (__VLS_ctx.selectedRegions),
    options: __VLS_ctx.pcaTextArr,
    placeholder: "请选择省市区",
}));
const __VLS_693 = __VLS_692({
    ...{ 'onChange': {} },
    modelValue: (__VLS_ctx.selectedRegions),
    options: __VLS_ctx.pcaTextArr,
    placeholder: "请选择省市区",
}, ...__VLS_functionalComponentArgsRest(__VLS_692));
let __VLS_696;
const __VLS_697 = ({ change: {} },
    { onChange: (__VLS_ctx.onSelectRegionChange) });
var __VLS_694;
var __VLS_695;
// @ts-ignore
[selectedRegions, pcaTextArr, onSelectRegionChange,];
var __VLS_688;
let __VLS_698;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_699 = __VLS_asFunctionalComponent1(__VLS_698, new __VLS_698({
    label: "详细地址：",
}));
const __VLS_700 = __VLS_699({
    label: "详细地址：",
}, ...__VLS_functionalComponentArgsRest(__VLS_699));
const { default: __VLS_703 } = __VLS_701.slots;
let __VLS_704;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_705 = __VLS_asFunctionalComponent1(__VLS_704, new __VLS_704({
    modelValue: (__VLS_ctx.receiverInfo.receiverDetailAddress),
    type: "textarea",
    rows: (3),
}));
const __VLS_706 = __VLS_705({
    modelValue: (__VLS_ctx.receiverInfo.receiverDetailAddress),
    type: "textarea",
    rows: (3),
}, ...__VLS_functionalComponentArgsRest(__VLS_705));
// @ts-ignore
[receiverInfo,];
var __VLS_701;
// @ts-ignore
[];
var __VLS_647;
{
    const { footer: __VLS_709 } = __VLS_641.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_710;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_711 = __VLS_asFunctionalComponent1(__VLS_710, new __VLS_710({
        ...{ 'onClick': {} },
    }));
    const __VLS_712 = __VLS_711({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_711));
    let __VLS_715;
    const __VLS_716 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.receiverDialogVisible = false;
                // @ts-ignore
                [receiverDialogVisible,];
            } });
    const { default: __VLS_717 } = __VLS_713.slots;
    // @ts-ignore
    [];
    var __VLS_713;
    var __VLS_714;
    let __VLS_718;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_719 = __VLS_asFunctionalComponent1(__VLS_718, new __VLS_718({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_720 = __VLS_719({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_719));
    let __VLS_723;
    const __VLS_724 = ({ click: {} },
        { onClick: (__VLS_ctx.handleUpdateReceiverInfo) });
    const { default: __VLS_725 } = __VLS_721.slots;
    // @ts-ignore
    [handleUpdateReceiverInfo,];
    var __VLS_721;
    var __VLS_722;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_641;
let __VLS_726;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_727 = __VLS_asFunctionalComponent1(__VLS_726, new __VLS_726({
    title: "修改费用信息",
    modelValue: (__VLS_ctx.moneyDialogVisible),
    width: "40%",
}));
const __VLS_728 = __VLS_727({
    title: "修改费用信息",
    modelValue: (__VLS_ctx.moneyDialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_727));
const { default: __VLS_731 } = __VLS_729.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-layout" },
});
/** @type {__VLS_StyleScopedClasses['table-layout']} */ ;
let __VLS_732;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_733 = __VLS_asFunctionalComponent1(__VLS_732, new __VLS_732({}));
const __VLS_734 = __VLS_733({}, ...__VLS_functionalComponentArgsRest(__VLS_733));
const { default: __VLS_737 } = __VLS_735.slots;
let __VLS_738;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_739 = __VLS_asFunctionalComponent1(__VLS_738, new __VLS_738({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_740 = __VLS_739({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_739));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_743 } = __VLS_741.slots;
// @ts-ignore
[moneyDialogVisible,];
var __VLS_741;
let __VLS_744;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_745 = __VLS_asFunctionalComponent1(__VLS_744, new __VLS_744({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_746 = __VLS_745({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_745));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_749 } = __VLS_747.slots;
// @ts-ignore
[];
var __VLS_747;
let __VLS_750;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_751 = __VLS_asFunctionalComponent1(__VLS_750, new __VLS_750({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_752 = __VLS_751({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_751));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_755 } = __VLS_753.slots;
// @ts-ignore
[];
var __VLS_753;
let __VLS_756;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_757 = __VLS_asFunctionalComponent1(__VLS_756, new __VLS_756({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_758 = __VLS_757({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_757));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_761 } = __VLS_759.slots;
// @ts-ignore
[];
var __VLS_759;
// @ts-ignore
[];
var __VLS_735;
let __VLS_762;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_763 = __VLS_asFunctionalComponent1(__VLS_762, new __VLS_762({}));
const __VLS_764 = __VLS_763({}, ...__VLS_functionalComponentArgsRest(__VLS_763));
const { default: __VLS_767 } = __VLS_765.slots;
let __VLS_768;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_769 = __VLS_asFunctionalComponent1(__VLS_768, new __VLS_768({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_770 = __VLS_769({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_769));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_773 } = __VLS_771.slots;
(__VLS_ctx.order.totalAmount);
// @ts-ignore
[order,];
var __VLS_771;
let __VLS_774;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_775 = __VLS_asFunctionalComponent1(__VLS_774, new __VLS_774({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_776 = __VLS_775({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_775));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_779 } = __VLS_777.slots;
let __VLS_780;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_781 = __VLS_asFunctionalComponent1(__VLS_780, new __VLS_780({
    modelValue: (__VLS_ctx.moneyInfo.freightAmount),
    modelModifiers: { number: true, },
    size: "small",
}));
const __VLS_782 = __VLS_781({
    modelValue: (__VLS_ctx.moneyInfo.freightAmount),
    modelModifiers: { number: true, },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_781));
const { default: __VLS_785 } = __VLS_783.slots;
{
    const { prepend: __VLS_786 } = __VLS_783.slots;
    // @ts-ignore
    [moneyInfo,];
}
// @ts-ignore
[];
var __VLS_783;
// @ts-ignore
[];
var __VLS_777;
let __VLS_787;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_788 = __VLS_asFunctionalComponent1(__VLS_787, new __VLS_787({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_789 = __VLS_788({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_788));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_792 } = __VLS_790.slots;
(__VLS_ctx.order.couponAmount);
// @ts-ignore
[order,];
var __VLS_790;
let __VLS_793;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_794 = __VLS_asFunctionalComponent1(__VLS_793, new __VLS_793({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_795 = __VLS_794({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_794));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_798 } = __VLS_796.slots;
(__VLS_ctx.order.integrationAmount);
// @ts-ignore
[order,];
var __VLS_796;
// @ts-ignore
[];
var __VLS_765;
let __VLS_799;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_800 = __VLS_asFunctionalComponent1(__VLS_799, new __VLS_799({}));
const __VLS_801 = __VLS_800({}, ...__VLS_functionalComponentArgsRest(__VLS_800));
const { default: __VLS_804 } = __VLS_802.slots;
let __VLS_805;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_806 = __VLS_asFunctionalComponent1(__VLS_805, new __VLS_805({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_807 = __VLS_806({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_806));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_810 } = __VLS_808.slots;
// @ts-ignore
[];
var __VLS_808;
let __VLS_811;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_812 = __VLS_asFunctionalComponent1(__VLS_811, new __VLS_811({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_813 = __VLS_812({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_812));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_816 } = __VLS_814.slots;
// @ts-ignore
[];
var __VLS_814;
let __VLS_817;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_818 = __VLS_asFunctionalComponent1(__VLS_817, new __VLS_817({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_819 = __VLS_818({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_818));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_822 } = __VLS_820.slots;
// @ts-ignore
[];
var __VLS_820;
let __VLS_823;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_824 = __VLS_asFunctionalComponent1(__VLS_823, new __VLS_823({
    span: (6),
    ...{ class: "table-cell-title" },
}));
const __VLS_825 = __VLS_824({
    span: (6),
    ...{ class: "table-cell-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_824));
/** @type {__VLS_StyleScopedClasses['table-cell-title']} */ ;
const { default: __VLS_828 } = __VLS_826.slots;
// @ts-ignore
[];
var __VLS_826;
// @ts-ignore
[];
var __VLS_802;
let __VLS_829;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_830 = __VLS_asFunctionalComponent1(__VLS_829, new __VLS_829({}));
const __VLS_831 = __VLS_830({}, ...__VLS_functionalComponentArgsRest(__VLS_830));
const { default: __VLS_834 } = __VLS_832.slots;
let __VLS_835;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_836 = __VLS_asFunctionalComponent1(__VLS_835, new __VLS_835({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_837 = __VLS_836({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_836));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_840 } = __VLS_838.slots;
(__VLS_ctx.order.promotionAmount);
// @ts-ignore
[order,];
var __VLS_838;
let __VLS_841;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_842 = __VLS_asFunctionalComponent1(__VLS_841, new __VLS_841({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_843 = __VLS_842({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_842));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_846 } = __VLS_844.slots;
let __VLS_847;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_848 = __VLS_asFunctionalComponent1(__VLS_847, new __VLS_847({
    modelValue: (__VLS_ctx.moneyInfo.discountAmount),
    modelModifiers: { number: true, },
    size: "small",
}));
const __VLS_849 = __VLS_848({
    modelValue: (__VLS_ctx.moneyInfo.discountAmount),
    modelModifiers: { number: true, },
    size: "small",
}, ...__VLS_functionalComponentArgsRest(__VLS_848));
const { default: __VLS_852 } = __VLS_850.slots;
{
    const { prepend: __VLS_853 } = __VLS_850.slots;
    // @ts-ignore
    [moneyInfo,];
}
// @ts-ignore
[];
var __VLS_850;
// @ts-ignore
[];
var __VLS_844;
let __VLS_854;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_855 = __VLS_asFunctionalComponent1(__VLS_854, new __VLS_854({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_856 = __VLS_855({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_855));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_859 } = __VLS_857.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
(__VLS_ctx.order.totalAmount + __VLS_ctx.moneyInfo.freightAmount);
// @ts-ignore
[order, moneyInfo,];
var __VLS_857;
let __VLS_860;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_861 = __VLS_asFunctionalComponent1(__VLS_860, new __VLS_860({
    span: (6),
    ...{ class: "table-cell" },
}));
const __VLS_862 = __VLS_861({
    span: (6),
    ...{ class: "table-cell" },
}, ...__VLS_functionalComponentArgsRest(__VLS_861));
/** @type {__VLS_StyleScopedClasses['table-cell']} */ ;
const { default: __VLS_865 } = __VLS_863.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
(__VLS_ctx.order.payAmount + __VLS_ctx.moneyInfo.freightAmount - __VLS_ctx.moneyInfo.discountAmount);
// @ts-ignore
[order, moneyInfo, moneyInfo,];
var __VLS_863;
// @ts-ignore
[];
var __VLS_832;
{
    const { footer: __VLS_866 } = __VLS_729.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_867;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_868 = __VLS_asFunctionalComponent1(__VLS_867, new __VLS_867({
        ...{ 'onClick': {} },
    }));
    const __VLS_869 = __VLS_868({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_868));
    let __VLS_872;
    const __VLS_873 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.moneyDialogVisible = false;
                // @ts-ignore
                [moneyDialogVisible,];
            } });
    const { default: __VLS_874 } = __VLS_870.slots;
    // @ts-ignore
    [];
    var __VLS_870;
    var __VLS_871;
    let __VLS_875;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_876 = __VLS_asFunctionalComponent1(__VLS_875, new __VLS_875({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_877 = __VLS_876({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_876));
    let __VLS_880;
    const __VLS_881 = ({ click: {} },
        { onClick: (__VLS_ctx.handleUpdateMoneyInfo) });
    const { default: __VLS_882 } = __VLS_878.slots;
    // @ts-ignore
    [handleUpdateMoneyInfo,];
    var __VLS_878;
    var __VLS_879;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_729;
let __VLS_883;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_884 = __VLS_asFunctionalComponent1(__VLS_883, new __VLS_883({
    title: "发送站内信",
    modelValue: (__VLS_ctx.messageDialogVisible),
    width: "40%",
}));
const __VLS_885 = __VLS_884({
    title: "发送站内信",
    modelValue: (__VLS_ctx.messageDialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_884));
const { default: __VLS_888 } = __VLS_886.slots;
let __VLS_889;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_890 = __VLS_asFunctionalComponent1(__VLS_889, new __VLS_889({
    model: (__VLS_ctx.message),
    ref: "receiverInfoForm",
    labelWidth: "150px",
}));
const __VLS_891 = __VLS_890({
    model: (__VLS_ctx.message),
    ref: "receiverInfoForm",
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_890));
var __VLS_894 = {};
const { default: __VLS_896 } = __VLS_892.slots;
let __VLS_897;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_898 = __VLS_asFunctionalComponent1(__VLS_897, new __VLS_897({
    label: "标题：",
}));
const __VLS_899 = __VLS_898({
    label: "标题：",
}, ...__VLS_functionalComponentArgsRest(__VLS_898));
const { default: __VLS_902 } = __VLS_900.slots;
let __VLS_903;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_904 = __VLS_asFunctionalComponent1(__VLS_903, new __VLS_903({
    modelValue: (__VLS_ctx.message.title),
    ...{ style: {} },
}));
const __VLS_905 = __VLS_904({
    modelValue: (__VLS_ctx.message.title),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_904));
// @ts-ignore
[messageDialogVisible, message, message,];
var __VLS_900;
let __VLS_908;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_909 = __VLS_asFunctionalComponent1(__VLS_908, new __VLS_908({
    label: "内容：",
}));
const __VLS_910 = __VLS_909({
    label: "内容：",
}, ...__VLS_functionalComponentArgsRest(__VLS_909));
const { default: __VLS_913 } = __VLS_911.slots;
let __VLS_914;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_915 = __VLS_asFunctionalComponent1(__VLS_914, new __VLS_914({
    modelValue: (__VLS_ctx.message.content),
    type: "textarea",
    rows: (3),
}));
const __VLS_916 = __VLS_915({
    modelValue: (__VLS_ctx.message.content),
    type: "textarea",
    rows: (3),
}, ...__VLS_functionalComponentArgsRest(__VLS_915));
// @ts-ignore
[message,];
var __VLS_911;
// @ts-ignore
[];
var __VLS_892;
{
    const { footer: __VLS_919 } = __VLS_886.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_920;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_921 = __VLS_asFunctionalComponent1(__VLS_920, new __VLS_920({
        ...{ 'onClick': {} },
    }));
    const __VLS_922 = __VLS_921({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_921));
    let __VLS_925;
    const __VLS_926 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.messageDialogVisible = false;
                // @ts-ignore
                [messageDialogVisible,];
            } });
    const { default: __VLS_927 } = __VLS_923.slots;
    // @ts-ignore
    [];
    var __VLS_923;
    var __VLS_924;
    let __VLS_928;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_929 = __VLS_asFunctionalComponent1(__VLS_928, new __VLS_928({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_930 = __VLS_929({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_929));
    let __VLS_933;
    const __VLS_934 = ({ click: {} },
        { onClick: (__VLS_ctx.handleSendMessage) });
    const { default: __VLS_935 } = __VLS_931.slots;
    // @ts-ignore
    [handleSendMessage,];
    var __VLS_931;
    var __VLS_932;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_886;
let __VLS_936;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_937 = __VLS_asFunctionalComponent1(__VLS_936, new __VLS_936({
    title: "关闭订单",
    visible: (__VLS_ctx.closeDialogVisible),
    width: "40%",
}));
const __VLS_938 = __VLS_937({
    title: "关闭订单",
    visible: (__VLS_ctx.closeDialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_937));
const { default: __VLS_941 } = __VLS_939.slots;
let __VLS_942;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_943 = __VLS_asFunctionalComponent1(__VLS_942, new __VLS_942({
    model: (__VLS_ctx.closeInfo),
    labelWidth: "150px",
}));
const __VLS_944 = __VLS_943({
    model: (__VLS_ctx.closeInfo),
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_943));
const { default: __VLS_947 } = __VLS_945.slots;
let __VLS_948;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_949 = __VLS_asFunctionalComponent1(__VLS_948, new __VLS_948({
    label: "操作备注：",
}));
const __VLS_950 = __VLS_949({
    label: "操作备注：",
}, ...__VLS_functionalComponentArgsRest(__VLS_949));
const { default: __VLS_953 } = __VLS_951.slots;
let __VLS_954;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_955 = __VLS_asFunctionalComponent1(__VLS_954, new __VLS_954({
    modelValue: (__VLS_ctx.closeInfo.note),
    type: "textarea",
    rows: (3),
}));
const __VLS_956 = __VLS_955({
    modelValue: (__VLS_ctx.closeInfo.note),
    type: "textarea",
    rows: (3),
}, ...__VLS_functionalComponentArgsRest(__VLS_955));
// @ts-ignore
[closeDialogVisible, closeInfo, closeInfo,];
var __VLS_951;
// @ts-ignore
[];
var __VLS_945;
{
    const { footer: __VLS_959 } = __VLS_939.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_960;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_961 = __VLS_asFunctionalComponent1(__VLS_960, new __VLS_960({
        ...{ 'onClick': {} },
    }));
    const __VLS_962 = __VLS_961({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_961));
    let __VLS_965;
    const __VLS_966 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.closeDialogVisible = false;
                // @ts-ignore
                [closeDialogVisible,];
            } });
    const { default: __VLS_967 } = __VLS_963.slots;
    // @ts-ignore
    [];
    var __VLS_963;
    var __VLS_964;
    let __VLS_968;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_969 = __VLS_asFunctionalComponent1(__VLS_968, new __VLS_968({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_970 = __VLS_969({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_969));
    let __VLS_973;
    const __VLS_974 = ({ click: {} },
        { onClick: (__VLS_ctx.handleCloseOrder) });
    const { default: __VLS_975 } = __VLS_971.slots;
    // @ts-ignore
    [handleCloseOrder,];
    var __VLS_971;
    var __VLS_972;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_939;
let __VLS_976;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_977 = __VLS_asFunctionalComponent1(__VLS_976, new __VLS_976({
    title: "备注订单",
    modelValue: (__VLS_ctx.markOrderDialogVisible),
    width: "40%",
}));
const __VLS_978 = __VLS_977({
    title: "备注订单",
    modelValue: (__VLS_ctx.markOrderDialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_977));
const { default: __VLS_981 } = __VLS_979.slots;
let __VLS_982;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_983 = __VLS_asFunctionalComponent1(__VLS_982, new __VLS_982({
    model: (__VLS_ctx.markInfo),
    labelWidth: "150px",
}));
const __VLS_984 = __VLS_983({
    model: (__VLS_ctx.markInfo),
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_983));
const { default: __VLS_987 } = __VLS_985.slots;
let __VLS_988;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_989 = __VLS_asFunctionalComponent1(__VLS_988, new __VLS_988({
    label: "操作备注：",
}));
const __VLS_990 = __VLS_989({
    label: "操作备注：",
}, ...__VLS_functionalComponentArgsRest(__VLS_989));
const { default: __VLS_993 } = __VLS_991.slots;
let __VLS_994;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_995 = __VLS_asFunctionalComponent1(__VLS_994, new __VLS_994({
    modelValue: (__VLS_ctx.markInfo.note),
    type: "textarea",
    rows: (3),
}));
const __VLS_996 = __VLS_995({
    modelValue: (__VLS_ctx.markInfo.note),
    type: "textarea",
    rows: (3),
}, ...__VLS_functionalComponentArgsRest(__VLS_995));
// @ts-ignore
[markOrderDialogVisible, markInfo, markInfo,];
var __VLS_991;
// @ts-ignore
[];
var __VLS_985;
{
    const { footer: __VLS_999 } = __VLS_979.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_1000;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_1001 = __VLS_asFunctionalComponent1(__VLS_1000, new __VLS_1000({
        ...{ 'onClick': {} },
    }));
    const __VLS_1002 = __VLS_1001({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_1001));
    let __VLS_1005;
    const __VLS_1006 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.markOrderDialogVisible = false;
                // @ts-ignore
                [markOrderDialogVisible,];
            } });
    const { default: __VLS_1007 } = __VLS_1003.slots;
    // @ts-ignore
    [];
    var __VLS_1003;
    var __VLS_1004;
    let __VLS_1008;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_1009 = __VLS_asFunctionalComponent1(__VLS_1008, new __VLS_1008({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_1010 = __VLS_1009({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_1009));
    let __VLS_1013;
    const __VLS_1014 = ({ click: {} },
        { onClick: (__VLS_ctx.handleMarkOrder) });
    const { default: __VLS_1015 } = __VLS_1011.slots;
    // @ts-ignore
    [handleMarkOrder,];
    var __VLS_1011;
    var __VLS_1012;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_979;
const __VLS_1016 = LogisticsDialog || LogisticsDialog;
// @ts-ignore
const __VLS_1017 = __VLS_asFunctionalComponent1(__VLS_1016, new __VLS_1016({
    modelValue: (__VLS_ctx.logisticsDialogVisible),
}));
const __VLS_1018 = __VLS_1017({
    modelValue: (__VLS_ctx.logisticsDialogVisible),
}, ...__VLS_functionalComponentArgsRest(__VLS_1017));
// @ts-ignore
var __VLS_414 = __VLS_413, __VLS_594 = __VLS_593, __VLS_650 = __VLS_649, __VLS_895 = __VLS_894;
// @ts-ignore
[logisticsDialogVisible,];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
