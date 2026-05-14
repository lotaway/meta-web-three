/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { Tickets } from '@element-plus/icons-vue';
import { getReturnReasonListAPI, returnReasonDeleteByIdsAPI, returnReasonUpdateStatusAPI, returnReasonCreateAPI, getReturnReasonByIdAPI, returnReasonUpdateAPI } from '@/apis/returnReason';
import { formatDateTime } from '@/utils/datetime';
// 列表查询参数
const listQuery = ref({
    pageNum: 1,
    pageSize: 10
});
// 列表数据
const list = ref([]);
// 列表总条数
const total = ref(0);
// 表格中被选中的行
const multipleSelection = ref([]);
// 表格数据加载进度条
const listLoading = ref(true);
// 获取列表数据
const getList = async () => {
    listLoading.value = true;
    const res = await getReturnReasonListAPI(listQuery.value);
    listLoading.value = false;
    list.value = res.data.list;
    total.value = res.data.total;
};
// 组件挂载后加载数据
onMounted(() => {
    getList();
});
// reason对象默认值
const defaultReturnReason = {
    name: '',
    sort: 0,
    status: 1,
};
// 编辑框是否可见
const dialogVisible = ref(false);
// 当前操作的reason对象
const returnReason = ref(Object.assign({}, defaultReturnReason));
// 当前操作的reasonId,为null时表示新增
const operateReasonId = ref();
// 批量操作类型
const operateType = ref();
// 所有批量操作
const operateOptions = ref([
    {
        label: "删除",
        value: 1
    }
]);
const handleAdd = () => {
    dialogVisible.value = true;
    operateReasonId.value = undefined;
    returnReason.value = Object.assign({}, defaultReturnReason);
};
const handleConfirm = async () => {
    if (!operateReasonId.value) {
        // 添加操作
        await returnReasonCreateAPI(returnReason.value);
        dialogVisible.value = false;
        operateReasonId.value = undefined;
        ElMessage({
            message: '添加成功！',
            type: 'success',
            duration: 1000
        });
        getList();
    }
    else {
        // 编辑操作
        await returnReasonUpdateAPI(operateReasonId.value, returnReason.value);
        dialogVisible.value = false;
        operateReasonId.value = undefined;
        ElMessage({
            message: '修改成功！',
            type: 'success',
            duration: 1000
        });
        getList();
    }
};
const handleUpdate = async (index, row) => {
    dialogVisible.value = true;
    operateReasonId.value = row.id;
    const res = await getReturnReasonByIdAPI(row.id);
    returnReason.value = res.data;
};
const handleDelete = (index, row) => {
    const ids = [];
    ids.push(row.id);
    deleteReasonMethod(ids);
};
const handleSelectionChange = (val) => {
    multipleSelection.value = val;
};
const handleStatusChange = async (index, row) => {
    const ids = [];
    ids.push(row.id);
    await returnReasonUpdateStatusAPI({ ids: ids.join(','), status: row.status });
    ElMessage({
        message: '状态修改成功',
        type: 'success'
    });
};
const handleBatchOperate = () => {
    if (!multipleSelection.value || multipleSelection.value.length < 1) {
        ElMessage({
            message: '请选择要操作的条目',
            type: 'warning',
            duration: 1000
        });
        return;
    }
    if (operateType.value === 1) {
        deleteReasonMethod(multipleSelection.value.map(item => item.id));
    }
};
const handleSizeChange = (val) => {
    listQuery.value.pageNum = 1;
    listQuery.value.pageSize = val;
    getList();
};
const handleCurrentChange = (val) => {
    listQuery.value.pageNum = val;
    getList();
};
const deleteReasonMethod = async (ids) => {
    await ElMessageBox.confirm('是否要进行该删除操作?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    await returnReasonDeleteByIdsAPI({ ids: ids.join(',') });
    ElMessage({
        message: '删除成功！',
        type: 'success',
        duration: 1000
    });
    listQuery.value.pageNum = 1;
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
let __VLS_17;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_18 = __VLS_asFunctionalComponent1(__VLS_17, new __VLS_17({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}));
const __VLS_19 = __VLS_18({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}, ...__VLS_functionalComponentArgsRest(__VLS_18));
let __VLS_22;
const __VLS_23 = ({ click: {} },
    { onClick: (__VLS_ctx.handleAdd) });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_24 } = __VLS_20.slots;
// @ts-ignore
[handleAdd,];
var __VLS_20;
var __VLS_21;
// @ts-ignore
[];
var __VLS_3;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_25;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_26 = __VLS_asFunctionalComponent1(__VLS_25, new __VLS_25({
    ...{ 'onSelectionChange': {} },
    ref: "returnReasonTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_27 = __VLS_26({
    ...{ 'onSelectionChange': {} },
    ref: "returnReasonTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_26));
let __VLS_30;
const __VLS_31 = ({ selectionChange: {} },
    { onSelectionChange: (__VLS_ctx.handleSelectionChange) });
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_32 = {};
const { default: __VLS_34 } = __VLS_28.slots;
let __VLS_35;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_36 = __VLS_asFunctionalComponent1(__VLS_35, new __VLS_35({
    type: "selection",
    width: "60",
    align: "center",
}));
const __VLS_37 = __VLS_36({
    type: "selection",
    width: "60",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_36));
let __VLS_40;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_41 = __VLS_asFunctionalComponent1(__VLS_40, new __VLS_40({
    label: "编号",
    width: "80",
    align: "center",
}));
const __VLS_42 = __VLS_41({
    label: "编号",
    width: "80",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_41));
const { default: __VLS_45 } = __VLS_43.slots;
{
    const { default: __VLS_46 } = __VLS_43.slots;
    const [scope] = __VLS_vSlot(__VLS_46);
    (scope.row.id);
    // @ts-ignore
    [list, handleSelectionChange, vLoading, listLoading,];
    __VLS_43.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_43;
let __VLS_47;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({
    label: "原因类型",
    align: "center",
}));
const __VLS_49 = __VLS_48({
    label: "原因类型",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_48));
const { default: __VLS_52 } = __VLS_50.slots;
{
    const { default: __VLS_53 } = __VLS_50.slots;
    const [scope] = __VLS_vSlot(__VLS_53);
    (scope.row.name);
    // @ts-ignore
    [];
    __VLS_50.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_50;
let __VLS_54;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    label: "排序",
    width: "100",
    align: "center",
}));
const __VLS_56 = __VLS_55({
    label: "排序",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
const { default: __VLS_59 } = __VLS_57.slots;
{
    const { default: __VLS_60 } = __VLS_57.slots;
    const [scope] = __VLS_vSlot(__VLS_60);
    (scope.row.sort);
    // @ts-ignore
    [];
    __VLS_57.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_57;
let __VLS_61;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_62 = __VLS_asFunctionalComponent1(__VLS_61, new __VLS_61({
    label: "是否可用",
    align: "center",
}));
const __VLS_63 = __VLS_62({
    label: "是否可用",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_62));
const { default: __VLS_66 } = __VLS_64.slots;
{
    const { default: __VLS_67 } = __VLS_64.slots;
    const [scope] = __VLS_vSlot(__VLS_67);
    let __VLS_68;
    /** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
    elSwitch;
    // @ts-ignore
    const __VLS_69 = __VLS_asFunctionalComponent1(__VLS_68, new __VLS_68({
        ...{ 'onChange': {} },
        modelValue: (scope.row.status),
        activeValue: (1),
        inactiveValue: (0),
    }));
    const __VLS_70 = __VLS_69({
        ...{ 'onChange': {} },
        modelValue: (scope.row.status),
        activeValue: (1),
        inactiveValue: (0),
    }, ...__VLS_functionalComponentArgsRest(__VLS_69));
    let __VLS_73;
    const __VLS_74 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handleStatusChange(scope.$index, scope.row);
                // @ts-ignore
                [handleStatusChange,];
            } });
    var __VLS_71;
    var __VLS_72;
    // @ts-ignore
    [];
    __VLS_64.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_64;
let __VLS_75;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_76 = __VLS_asFunctionalComponent1(__VLS_75, new __VLS_75({
    label: "添加时间",
    width: "180",
    align: "center",
}));
const __VLS_77 = __VLS_76({
    label: "添加时间",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_76));
const { default: __VLS_80 } = __VLS_78.slots;
{
    const { default: __VLS_81 } = __VLS_78.slots;
    const [scope] = __VLS_vSlot(__VLS_81);
    (__VLS_ctx.formatDateTime(scope.row.createTime));
    // @ts-ignore
    [formatDateTime,];
    __VLS_78.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_78;
let __VLS_82;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_83 = __VLS_asFunctionalComponent1(__VLS_82, new __VLS_82({
    label: "操作",
    width: "160",
    align: "center",
}));
const __VLS_84 = __VLS_83({
    label: "操作",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_83));
const { default: __VLS_87 } = __VLS_85.slots;
{
    const { default: __VLS_88 } = __VLS_85.slots;
    const [scope] = __VLS_vSlot(__VLS_88);
    let __VLS_89;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_90 = __VLS_asFunctionalComponent1(__VLS_89, new __VLS_89({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_91 = __VLS_90({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_90));
    let __VLS_94;
    const __VLS_95 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdate(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdate,];
            } });
    const { default: __VLS_96 } = __VLS_92.slots;
    // @ts-ignore
    [];
    var __VLS_92;
    var __VLS_93;
    let __VLS_97;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_98 = __VLS_asFunctionalComponent1(__VLS_97, new __VLS_97({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_99 = __VLS_98({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_98));
    let __VLS_102;
    const __VLS_103 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_104 } = __VLS_100.slots;
    // @ts-ignore
    [];
    var __VLS_100;
    var __VLS_101;
    // @ts-ignore
    [];
    __VLS_85.slots['' /* empty slot name completion */];
}
// @ts-ignore
[];
var __VLS_85;
// @ts-ignore
[];
var __VLS_28;
var __VLS_29;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "batch-operate-container" },
});
/** @type {__VLS_StyleScopedClasses['batch-operate-container']} */ ;
let __VLS_105;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_106 = __VLS_asFunctionalComponent1(__VLS_105, new __VLS_105({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}));
const __VLS_107 = __VLS_106({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}, ...__VLS_functionalComponentArgsRest(__VLS_106));
const { default: __VLS_110 } = __VLS_108.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.operateOptions))) {
    let __VLS_111;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_112 = __VLS_asFunctionalComponent1(__VLS_111, new __VLS_111({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_113 = __VLS_112({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_112));
    // @ts-ignore
    [operateType, operateOptions,];
}
// @ts-ignore
[];
var __VLS_108;
let __VLS_116;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_117 = __VLS_asFunctionalComponent1(__VLS_116, new __VLS_116({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}));
const __VLS_118 = __VLS_117({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_117));
let __VLS_121;
const __VLS_122 = ({ click: {} },
    { onClick: (__VLS_ctx.handleBatchOperate) });
/** @type {__VLS_StyleScopedClasses['search-button']} */ ;
const { default: __VLS_123 } = __VLS_119.slots;
// @ts-ignore
[handleBatchOperate,];
var __VLS_119;
var __VLS_120;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_124;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_125 = __VLS_asFunctionalComponent1(__VLS_124, new __VLS_124({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}));
const __VLS_126 = __VLS_125({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_125));
let __VLS_129;
const __VLS_130 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_131 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_127;
var __VLS_128;
let __VLS_132;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_133 = __VLS_asFunctionalComponent1(__VLS_132, new __VLS_132({
    title: "添加退货原因",
    modelValue: (__VLS_ctx.dialogVisible),
    width: "30%",
}));
const __VLS_134 = __VLS_133({
    title: "添加退货原因",
    modelValue: (__VLS_ctx.dialogVisible),
    width: "30%",
}, ...__VLS_functionalComponentArgsRest(__VLS_133));
const { default: __VLS_137 } = __VLS_135.slots;
let __VLS_138;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_139 = __VLS_asFunctionalComponent1(__VLS_138, new __VLS_138({
    model: (__VLS_ctx.returnReason),
    ref: "reasonForm",
    labelWidth: "150px",
}));
const __VLS_140 = __VLS_139({
    model: (__VLS_ctx.returnReason),
    ref: "reasonForm",
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_139));
var __VLS_143 = {};
const { default: __VLS_145 } = __VLS_141.slots;
let __VLS_146;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_147 = __VLS_asFunctionalComponent1(__VLS_146, new __VLS_146({
    label: "原因类型：",
}));
const __VLS_148 = __VLS_147({
    label: "原因类型：",
}, ...__VLS_functionalComponentArgsRest(__VLS_147));
const { default: __VLS_151 } = __VLS_149.slots;
let __VLS_152;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_153 = __VLS_asFunctionalComponent1(__VLS_152, new __VLS_152({
    modelValue: (__VLS_ctx.returnReason.name),
    ...{ class: "input-width" },
}));
const __VLS_154 = __VLS_153({
    modelValue: (__VLS_ctx.returnReason.name),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_153));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery, listQuery, total, handleSizeChange, handleCurrentChange, dialogVisible, returnReason, returnReason,];
var __VLS_149;
let __VLS_157;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_158 = __VLS_asFunctionalComponent1(__VLS_157, new __VLS_157({
    label: "排序：",
}));
const __VLS_159 = __VLS_158({
    label: "排序：",
}, ...__VLS_functionalComponentArgsRest(__VLS_158));
const { default: __VLS_162 } = __VLS_160.slots;
let __VLS_163;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_164 = __VLS_asFunctionalComponent1(__VLS_163, new __VLS_163({
    modelValue: (__VLS_ctx.returnReason.sort),
    ...{ class: "input-width" },
}));
const __VLS_165 = __VLS_164({
    modelValue: (__VLS_ctx.returnReason.sort),
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_164));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[returnReason,];
var __VLS_160;
let __VLS_168;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_169 = __VLS_asFunctionalComponent1(__VLS_168, new __VLS_168({
    label: "是否启用：",
}));
const __VLS_170 = __VLS_169({
    label: "是否启用：",
}, ...__VLS_functionalComponentArgsRest(__VLS_169));
const { default: __VLS_173 } = __VLS_171.slots;
let __VLS_174;
/** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
elSwitch;
// @ts-ignore
const __VLS_175 = __VLS_asFunctionalComponent1(__VLS_174, new __VLS_174({
    modelValue: (__VLS_ctx.returnReason.status),
    activeValue: (1),
    inactiveValue: (0),
}));
const __VLS_176 = __VLS_175({
    modelValue: (__VLS_ctx.returnReason.status),
    activeValue: (1),
    inactiveValue: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_175));
// @ts-ignore
[returnReason,];
var __VLS_171;
// @ts-ignore
[];
var __VLS_141;
{
    const { footer: __VLS_179 } = __VLS_135.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_180;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_181 = __VLS_asFunctionalComponent1(__VLS_180, new __VLS_180({
        ...{ 'onClick': {} },
    }));
    const __VLS_182 = __VLS_181({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_181));
    let __VLS_185;
    const __VLS_186 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.dialogVisible = false;
                // @ts-ignore
                [dialogVisible,];
            } });
    const { default: __VLS_187 } = __VLS_183.slots;
    // @ts-ignore
    [];
    var __VLS_183;
    var __VLS_184;
    let __VLS_188;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_189 = __VLS_asFunctionalComponent1(__VLS_188, new __VLS_188({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_190 = __VLS_189({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_189));
    let __VLS_193;
    const __VLS_194 = ({ click: {} },
        { onClick: (__VLS_ctx.handleConfirm) });
    const { default: __VLS_195 } = __VLS_191.slots;
    // @ts-ignore
    [handleConfirm,];
    var __VLS_191;
    var __VLS_192;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_135;
// @ts-ignore
var __VLS_33 = __VLS_32, __VLS_144 = __VLS_143;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
