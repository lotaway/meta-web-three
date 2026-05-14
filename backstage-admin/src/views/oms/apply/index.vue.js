/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { Search, Tickets } from '@element-plus/icons-vue';
import { formatDateTime } from '@/utils/datetime';
import { getReturnApplyListAPI, returnApplyDeleteByIdsAPI } from '@/apis/returnApply';
// 获取路由对象
const router = useRouter();
// 默认处理状态
const defaultStatusOptions = [
    {
        label: '待处理',
        value: 0
    },
    {
        label: '退货中',
        value: 1
    },
    {
        label: '已完成',
        value: 2
    },
    {
        label: '已拒绝',
        value: 3
    }
];
// 列表查询参数
const listQuery = ref({
    pageNum: 1,
    pageSize: 10
});
// 状态选项
const statusOptions = ref(Object.assign({}, defaultStatusOptions));
// 列表数据
const list = ref([]);
// 总数
const total = ref(0);
// 加载状态
const listLoading = ref(false);
// 多选数据
const multipleSelection = ref([]);
// 获取列表
const getList = async () => {
    listLoading.value = true;
    const res = await getReturnApplyListAPI(listQuery.value);
    listLoading.value = false;
    list.value = res.data.list;
    total.value = res.data.total;
};
// 组件挂载后获取列表
onMounted(() => {
    getList();
});
// 操作类型
const operateType = ref();
// 操作选项
const operateOptions = ref([
    {
        label: "批量删除",
        value: 1
    }
]);
// 格式化状态
const formatStatus = (status) => {
    return defaultStatusOptions.find(item => item.value === status)?.label;
};
// 格式化退款金额
const formatReturnAmount = (row) => {
    return row.productRealPrice * row.productCount;
};
// 处理选择变化
const handleSelectionChange = (val) => {
    multipleSelection.value = val;
};
// 重置搜索
const handleResetSearch = () => {
    listQuery.value = { pageNum: 1, pageSize: 10 };
};
// 搜索列表
const handleSearchList = () => {
    listQuery.value.pageNum = 1;
    getList();
};
// 查看详情
const handleViewDetail = (index, row) => {
    router.push({ path: '/oms/returnApplyDetail', query: { id: row.id } });
};
// 批量操作
const handleBatchOperate = async () => {
    if (!multipleSelection.value || multipleSelection.value.length < 1) {
        ElMessage({
            message: '请选择要操作的申请',
            type: 'warning',
            duration: 1000
        });
        return;
    }
    if (operateType.value === 1) {
        // 批量删除
        await ElMessageBox.confirm('是否要进行删除操作?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await returnApplyDeleteByIdsAPI({ ids: multipleSelection.value.map(item => item.id).join(',') });
        getList();
        ElMessage({
            type: 'success',
            message: '删除成功!'
        });
    }
};
// 处理每页大小变化
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
    modelValue: (__VLS_ctx.listQuery.id),
    ...{ class: "input-width" },
    placeholder: "服务单号",
}));
const __VLS_47 = __VLS_46({
    modelValue: (__VLS_ctx.listQuery.id),
    ...{ class: "input-width" },
    placeholder: "服务单号",
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
    label: "处理状态：",
}));
const __VLS_52 = __VLS_51({
    label: "处理状态：",
}, ...__VLS_functionalComponentArgsRest(__VLS_51));
const { default: __VLS_55 } = __VLS_53.slots;
let __VLS_56;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
    modelValue: (__VLS_ctx.listQuery.status),
    placeholder: "全部",
    clearable: true,
    ...{ class: "input-width" },
}));
const __VLS_58 = __VLS_57({
    modelValue: (__VLS_ctx.listQuery.status),
    placeholder: "全部",
    clearable: true,
    ...{ class: "input-width" },
}, ...__VLS_functionalComponentArgsRest(__VLS_57));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_61 } = __VLS_59.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.statusOptions))) {
    let __VLS_62;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_63 = __VLS_asFunctionalComponent1(__VLS_62, new __VLS_62({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_64 = __VLS_63({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_63));
    // @ts-ignore
    [listQuery, statusOptions,];
}
// @ts-ignore
[];
var __VLS_59;
// @ts-ignore
[];
var __VLS_53;
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    label: "申请时间：",
}));
const __VLS_69 = __VLS_68({
    label: "申请时间：",
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
const { default: __VLS_72 } = __VLS_70.slots;
let __VLS_73;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_74 = __VLS_asFunctionalComponent1(__VLS_73, new __VLS_73({
    ...{ class: "input-width" },
    modelValue: (__VLS_ctx.listQuery.createTime),
    valueFormat: "yyyy-MM-dd",
    type: "date",
    placeholder: "请选择时间",
}));
const __VLS_75 = __VLS_74({
    ...{ class: "input-width" },
    modelValue: (__VLS_ctx.listQuery.createTime),
    valueFormat: "yyyy-MM-dd",
    type: "date",
    placeholder: "请选择时间",
}, ...__VLS_functionalComponentArgsRest(__VLS_74));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery,];
var __VLS_70;
let __VLS_78;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_79 = __VLS_asFunctionalComponent1(__VLS_78, new __VLS_78({
    label: "操作人员：",
}));
const __VLS_80 = __VLS_79({
    label: "操作人员：",
}, ...__VLS_functionalComponentArgsRest(__VLS_79));
const { default: __VLS_83 } = __VLS_81.slots;
let __VLS_84;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_85 = __VLS_asFunctionalComponent1(__VLS_84, new __VLS_84({
    modelValue: (__VLS_ctx.listQuery.handleMan),
    ...{ class: "input-width" },
    placeholder: "全部",
}));
const __VLS_86 = __VLS_85({
    modelValue: (__VLS_ctx.listQuery.handleMan),
    ...{ class: "input-width" },
    placeholder: "全部",
}, ...__VLS_functionalComponentArgsRest(__VLS_85));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery,];
var __VLS_81;
let __VLS_89;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_90 = __VLS_asFunctionalComponent1(__VLS_89, new __VLS_89({
    label: "处理时间：",
}));
const __VLS_91 = __VLS_90({
    label: "处理时间：",
}, ...__VLS_functionalComponentArgsRest(__VLS_90));
const { default: __VLS_94 } = __VLS_92.slots;
let __VLS_95;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_96 = __VLS_asFunctionalComponent1(__VLS_95, new __VLS_95({
    ...{ class: "input-width" },
    modelValue: (__VLS_ctx.listQuery.handleTime),
    valueFormat: "yyyy-MM-dd",
    type: "date",
    placeholder: "请选择时间",
}));
const __VLS_97 = __VLS_96({
    ...{ class: "input-width" },
    modelValue: (__VLS_ctx.listQuery.handleTime),
    valueFormat: "yyyy-MM-dd",
    type: "date",
    placeholder: "请选择时间",
}, ...__VLS_functionalComponentArgsRest(__VLS_96));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery,];
var __VLS_92;
// @ts-ignore
[];
var __VLS_36;
// @ts-ignore
[];
var __VLS_3;
let __VLS_100;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_101 = __VLS_asFunctionalComponent1(__VLS_100, new __VLS_100({
    ...{ class: "operate-container" },
    shadow: "never",
}));
const __VLS_102 = __VLS_101({
    ...{ class: "operate-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_101));
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
const { default: __VLS_105 } = __VLS_103.slots;
let __VLS_106;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_107 = __VLS_asFunctionalComponent1(__VLS_106, new __VLS_106({
    ...{ class: "el-icon-middle" },
}));
const __VLS_108 = __VLS_107({
    ...{ class: "el-icon-middle" },
}, ...__VLS_functionalComponentArgsRest(__VLS_107));
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_111 } = __VLS_109.slots;
let __VLS_112;
/** @ts-ignore @type { | typeof __VLS_components.Tickets} */
Tickets;
// @ts-ignore
const __VLS_113 = __VLS_asFunctionalComponent1(__VLS_112, new __VLS_112({}));
const __VLS_114 = __VLS_113({}, ...__VLS_functionalComponentArgsRest(__VLS_113));
// @ts-ignore
[];
var __VLS_109;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
// @ts-ignore
[];
var __VLS_103;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_117;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_118 = __VLS_asFunctionalComponent1(__VLS_117, new __VLS_117({
    ...{ 'onSelectionChange': {} },
    ref: "returnApplyTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_119 = __VLS_118({
    ...{ 'onSelectionChange': {} },
    ref: "returnApplyTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_118));
let __VLS_122;
const __VLS_123 = ({ selectionChange: {} },
    { onSelectionChange: (__VLS_ctx.handleSelectionChange) });
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_124 = {};
const { default: __VLS_126 } = __VLS_120.slots;
let __VLS_127;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_128 = __VLS_asFunctionalComponent1(__VLS_127, new __VLS_127({
    type: "selection",
    width: "60",
    align: "center",
}));
const __VLS_129 = __VLS_128({
    type: "selection",
    width: "60",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_128));
let __VLS_132;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_133 = __VLS_asFunctionalComponent1(__VLS_132, new __VLS_132({
    label: "服务单号",
    width: "180",
    align: "center",
}));
const __VLS_134 = __VLS_133({
    label: "服务单号",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_133));
const { default: __VLS_137 } = __VLS_135.slots;
{
    const { default: __VLS_138 } = __VLS_135.slots;
    const [scope] = __VLS_vSlot(__VLS_138);
    (scope.row.id);
    // @ts-ignore
    [list, handleSelectionChange, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_135;
let __VLS_139;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_140 = __VLS_asFunctionalComponent1(__VLS_139, new __VLS_139({
    label: "申请时间",
    width: "180",
    align: "center",
}));
const __VLS_141 = __VLS_140({
    label: "申请时间",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_140));
const { default: __VLS_144 } = __VLS_142.slots;
{
    const { default: __VLS_145 } = __VLS_142.slots;
    const [scope] = __VLS_vSlot(__VLS_145);
    (__VLS_ctx.formatDateTime(scope.row.createTime));
    // @ts-ignore
    [formatDateTime,];
}
// @ts-ignore
[];
var __VLS_142;
let __VLS_146;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_147 = __VLS_asFunctionalComponent1(__VLS_146, new __VLS_146({
    label: "用户账号",
    align: "center",
}));
const __VLS_148 = __VLS_147({
    label: "用户账号",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_147));
const { default: __VLS_151 } = __VLS_149.slots;
{
    const { default: __VLS_152 } = __VLS_149.slots;
    const [scope] = __VLS_vSlot(__VLS_152);
    (scope.row.memberUsername);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_149;
let __VLS_153;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_154 = __VLS_asFunctionalComponent1(__VLS_153, new __VLS_153({
    label: "退款金额",
    width: "180",
    align: "center",
}));
const __VLS_155 = __VLS_154({
    label: "退款金额",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_154));
const { default: __VLS_158 } = __VLS_156.slots;
{
    const { default: __VLS_159 } = __VLS_156.slots;
    const [scope] = __VLS_vSlot(__VLS_159);
    (__VLS_ctx.formatReturnAmount(scope.row));
    // @ts-ignore
    [formatReturnAmount,];
}
// @ts-ignore
[];
var __VLS_156;
let __VLS_160;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_161 = __VLS_asFunctionalComponent1(__VLS_160, new __VLS_160({
    label: "申请状态",
    width: "180",
    align: "center",
}));
const __VLS_162 = __VLS_161({
    label: "申请状态",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_161));
const { default: __VLS_165 } = __VLS_163.slots;
{
    const { default: __VLS_166 } = __VLS_163.slots;
    const [scope] = __VLS_vSlot(__VLS_166);
    (__VLS_ctx.formatStatus(scope.row.status));
    // @ts-ignore
    [formatStatus,];
}
// @ts-ignore
[];
var __VLS_163;
let __VLS_167;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_168 = __VLS_asFunctionalComponent1(__VLS_167, new __VLS_167({
    label: "处理时间",
    width: "180",
    align: "center",
}));
const __VLS_169 = __VLS_168({
    label: "处理时间",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_168));
const { default: __VLS_172 } = __VLS_170.slots;
{
    const { default: __VLS_173 } = __VLS_170.slots;
    const [scope] = __VLS_vSlot(__VLS_173);
    (__VLS_ctx.formatDateTime(scope.row.handleTime));
    // @ts-ignore
    [formatDateTime,];
}
// @ts-ignore
[];
var __VLS_170;
let __VLS_174;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_175 = __VLS_asFunctionalComponent1(__VLS_174, new __VLS_174({
    label: "操作",
    width: "180",
    align: "center",
}));
const __VLS_176 = __VLS_175({
    label: "操作",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_175));
const { default: __VLS_179 } = __VLS_177.slots;
{
    const { default: __VLS_180 } = __VLS_177.slots;
    const [scope] = __VLS_vSlot(__VLS_180);
    let __VLS_181;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_182 = __VLS_asFunctionalComponent1(__VLS_181, new __VLS_181({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_183 = __VLS_182({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_182));
    let __VLS_186;
    const __VLS_187 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleViewDetail(scope.$index, scope.row);
                // @ts-ignore
                [handleViewDetail,];
            } });
    const { default: __VLS_188 } = __VLS_184.slots;
    // @ts-ignore
    [];
    var __VLS_184;
    var __VLS_185;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_177;
// @ts-ignore
[];
var __VLS_120;
var __VLS_121;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "batch-operate-container" },
});
/** @type {__VLS_StyleScopedClasses['batch-operate-container']} */ ;
let __VLS_189;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_190 = __VLS_asFunctionalComponent1(__VLS_189, new __VLS_189({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}));
const __VLS_191 = __VLS_190({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}, ...__VLS_functionalComponentArgsRest(__VLS_190));
const { default: __VLS_194 } = __VLS_192.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.operateOptions))) {
    let __VLS_195;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_196 = __VLS_asFunctionalComponent1(__VLS_195, new __VLS_195({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_197 = __VLS_196({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_196));
    // @ts-ignore
    [operateType, operateOptions,];
}
// @ts-ignore
[];
var __VLS_192;
let __VLS_200;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_201 = __VLS_asFunctionalComponent1(__VLS_200, new __VLS_200({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}));
const __VLS_202 = __VLS_201({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_201));
let __VLS_205;
const __VLS_206 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleBatchOperate();
            // @ts-ignore
            [handleBatchOperate,];
        } });
/** @type {__VLS_StyleScopedClasses['search-button']} */ ;
const { default: __VLS_207 } = __VLS_203.slots;
// @ts-ignore
[];
var __VLS_203;
var __VLS_204;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_208;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_209 = __VLS_asFunctionalComponent1(__VLS_208, new __VLS_208({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}));
const __VLS_210 = __VLS_209({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_209));
let __VLS_213;
const __VLS_214 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_215 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_211;
var __VLS_212;
// @ts-ignore
var __VLS_125 = __VLS_124;
// @ts-ignore
[listQuery, listQuery, total, handleSizeChange, handleCurrentChange,];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
