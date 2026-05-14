/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { Tickets } from '@element-plus/icons-vue';
import { orderUpdateDeliveryAPI } from '@/apis/order';
import { useOrderStore } from '@/stores/order';
// 获取路由对象
const router = useRouter();
// 获取 订单store
const orderStore = useOrderStore();
// 默认物流公司选项
const defaultLogisticsCompanies = ["顺丰快递", "圆通快递", "中通快递", "韵达快递"];
// 发货订单列表数据
const list = ref([]);
// 物流公司选项
const companyOptions = ref(defaultLogisticsCompanies);
// 根据订单对象获取详细地址
const fortmatAddress = (order) => {
    return order.receiverProvince + order.receiverCity + order.receiverRegion + order.receiverDetailAddress;
};
// 组件挂载后初始化数据
onMounted(() => {
    list.value = orderStore.deliverOrderList;
    // 清空store中的数据
    orderStore.setDeliverOrderList([]);
});
// 取消操作
const cancel = () => {
    router.back();
};
// 确认发货操作
const confirm = async () => {
    try {
        await ElMessageBox.confirm('是否要进行发货操作?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        const deliveryParamList = list.value.map(item => ({
            orderId: item.id,
            deliverySn: item.orderSn,
            deliveryCompany: item.deliveryCompany
        }));
        await orderUpdateDeliveryAPI(deliveryParamList);
        router.back();
        ElMessage({
            type: 'success',
            message: '发货成功!'
        });
    }
    catch (error) {
        if (error !== 'cancel') {
            ElMessage({
                type: 'info',
                message: '已取消发货'
            });
        }
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
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    ...{ class: "operate-container" },
    shadow: "never",
}));
const __VLS_2 = __VLS_1({
    ...{ class: "operate-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
const { default: __VLS_5 } = __VLS_3.slots;
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
/** @ts-ignore @type { | typeof __VLS_components.Tickets} */
Tickets;
// @ts-ignore
const __VLS_13 = __VLS_asFunctionalComponent1(__VLS_12, new __VLS_12({}));
const __VLS_14 = __VLS_13({}, ...__VLS_functionalComponentArgsRest(__VLS_13));
var __VLS_9;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
var __VLS_3;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_17;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_18 = __VLS_asFunctionalComponent1(__VLS_17, new __VLS_17({
    ref: "deliverOrderTable",
    ...{ style: {} },
    data: (__VLS_ctx.list),
    border: true,
}));
const __VLS_19 = __VLS_18({
    ref: "deliverOrderTable",
    ...{ style: {} },
    data: (__VLS_ctx.list),
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_18));
var __VLS_22 = {};
const { default: __VLS_24 } = __VLS_20.slots;
let __VLS_25;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_26 = __VLS_asFunctionalComponent1(__VLS_25, new __VLS_25({
    label: "订单编号",
    width: "180",
    align: "center",
}));
const __VLS_27 = __VLS_26({
    label: "订单编号",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_26));
const { default: __VLS_30 } = __VLS_28.slots;
{
    const { default: __VLS_31 } = __VLS_28.slots;
    const [scope] = __VLS_vSlot(__VLS_31);
    (scope.row.orderSn);
    // @ts-ignore
    [list,];
}
// @ts-ignore
[];
var __VLS_28;
let __VLS_32;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_33 = __VLS_asFunctionalComponent1(__VLS_32, new __VLS_32({
    label: "收货人",
    width: "150",
    align: "center",
}));
const __VLS_34 = __VLS_33({
    label: "收货人",
    width: "150",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_33));
const { default: __VLS_37 } = __VLS_35.slots;
{
    const { default: __VLS_38 } = __VLS_35.slots;
    const [scope] = __VLS_vSlot(__VLS_38);
    (scope.row.receiverName);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_35;
let __VLS_39;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_40 = __VLS_asFunctionalComponent1(__VLS_39, new __VLS_39({
    label: "手机号码",
    width: "160",
    align: "center",
}));
const __VLS_41 = __VLS_40({
    label: "手机号码",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_40));
const { default: __VLS_44 } = __VLS_42.slots;
{
    const { default: __VLS_45 } = __VLS_42.slots;
    const [scope] = __VLS_vSlot(__VLS_45);
    (scope.row.receiverPhone);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_42;
let __VLS_46;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_47 = __VLS_asFunctionalComponent1(__VLS_46, new __VLS_46({
    label: "邮政编码",
    width: "160",
    align: "center",
}));
const __VLS_48 = __VLS_47({
    label: "邮政编码",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_47));
const { default: __VLS_51 } = __VLS_49.slots;
{
    const { default: __VLS_52 } = __VLS_49.slots;
    const [scope] = __VLS_vSlot(__VLS_52);
    (scope.row.receiverPostCode);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_49;
let __VLS_53;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_54 = __VLS_asFunctionalComponent1(__VLS_53, new __VLS_53({
    label: "收货地址",
    align: "center",
}));
const __VLS_55 = __VLS_54({
    label: "收货地址",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_54));
const { default: __VLS_58 } = __VLS_56.slots;
{
    const { default: __VLS_59 } = __VLS_56.slots;
    const [scope] = __VLS_vSlot(__VLS_59);
    (__VLS_ctx.fortmatAddress(scope.row));
    // @ts-ignore
    [fortmatAddress,];
}
// @ts-ignore
[];
var __VLS_56;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    label: "配送方式",
    width: "200",
    align: "center",
}));
const __VLS_62 = __VLS_61({
    label: "配送方式",
    width: "200",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
const { default: __VLS_65 } = __VLS_63.slots;
{
    const { default: __VLS_66 } = __VLS_63.slots;
    const [scope] = __VLS_vSlot(__VLS_66);
    let __VLS_67;
    /** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
    elSelect;
    // @ts-ignore
    const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
        placeholder: "请选择物流公司",
        modelValue: (scope.row.deliveryCompany),
    }));
    const __VLS_69 = __VLS_68({
        placeholder: "请选择物流公司",
        modelValue: (scope.row.deliveryCompany),
    }, ...__VLS_functionalComponentArgsRest(__VLS_68));
    const { default: __VLS_72 } = __VLS_70.slots;
    for (const [item] of __VLS_vFor((__VLS_ctx.companyOptions))) {
        let __VLS_73;
        /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
        elOption;
        // @ts-ignore
        const __VLS_74 = __VLS_asFunctionalComponent1(__VLS_73, new __VLS_73({
            key: (item),
            label: (item),
            value: (item),
        }));
        const __VLS_75 = __VLS_74({
            key: (item),
            label: (item),
            value: (item),
        }, ...__VLS_functionalComponentArgsRest(__VLS_74));
        // @ts-ignore
        [companyOptions,];
    }
    // @ts-ignore
    [];
    var __VLS_70;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_63;
let __VLS_78;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_79 = __VLS_asFunctionalComponent1(__VLS_78, new __VLS_78({
    label: "物流单号",
    width: "180",
    align: "center",
}));
const __VLS_80 = __VLS_79({
    label: "物流单号",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_79));
const { default: __VLS_83 } = __VLS_81.slots;
{
    const { default: __VLS_84 } = __VLS_81.slots;
    const [scope] = __VLS_vSlot(__VLS_84);
    let __VLS_85;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_86 = __VLS_asFunctionalComponent1(__VLS_85, new __VLS_85({
        modelValue: (scope.row.deliverySn),
    }));
    const __VLS_87 = __VLS_86({
        modelValue: (scope.row.deliverySn),
    }, ...__VLS_functionalComponentArgsRest(__VLS_86));
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_81;
// @ts-ignore
[];
var __VLS_20;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_90;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_91 = __VLS_asFunctionalComponent1(__VLS_90, new __VLS_90({
    ...{ 'onClick': {} },
}));
const __VLS_92 = __VLS_91({
    ...{ 'onClick': {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_91));
let __VLS_95;
const __VLS_96 = ({ click: {} },
    { onClick: (__VLS_ctx.cancel) });
const { default: __VLS_97 } = __VLS_93.slots;
// @ts-ignore
[cancel,];
var __VLS_93;
var __VLS_94;
let __VLS_98;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_99 = __VLS_asFunctionalComponent1(__VLS_98, new __VLS_98({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_100 = __VLS_99({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_99));
let __VLS_103;
const __VLS_104 = ({ click: {} },
    { onClick: (__VLS_ctx.confirm) });
const { default: __VLS_105 } = __VLS_101.slots;
// @ts-ignore
[confirm,];
var __VLS_101;
var __VLS_102;
// @ts-ignore
var __VLS_23 = __VLS_22;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
