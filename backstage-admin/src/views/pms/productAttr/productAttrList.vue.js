/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getProductAttributeListAPI, productAttributeDeleteByIds } from '@/apis/productAttr';
import { Tickets } from '@element-plus/icons-vue';
// 获取路由对象
const router = useRouter();
// 属性分类ID
const cid = Number(router.currentRoute.value.query.cid);
// 属性类型
const type = Number(router.currentRoute.value.query.type);
// 列表查询数据
const listQuery = ref({
    pageNum: 1,
    pageSize: 5,
    type: type
});
const list = ref([]);
// 列表数据
const total = ref(0);
// 加载状态
const listLoading = ref(true);
// 获取列表数据
const getList = async () => {
    listLoading.value = true;
    try {
        const response = await getProductAttributeListAPI(cid, listQuery.value);
        listLoading.value = false;
        list.value = response.data.list;
        total.value = response.data.total;
    }
    catch (error) {
        listLoading.value = false;
        console.error(error);
    }
};
// 组件挂载后执行
onMounted(() => {
    getList();
});
// 当前批量操作类型
const operateType = ref();
// 批量操作选中条目
const multipleSelection = ref([]);
// 批量操作类型集合
const operates = ref([
    {
        label: "删除",
        value: "deleteProductAttr"
    }
]);
// 添加产品属性
const addProductAttr = () => {
    router.push({ path: '/pms/addProductAttr', query: { cid: router.currentRoute.value.query.cid, type: router.currentRoute.value.query.type } });
};
// 处理表格选中状态变化
const handleSelectionChange = (val) => {
    multipleSelection.value = val;
};
// 处理批量操作
const handleBatchOperate = async () => {
    if (multipleSelection.value.length < 1) {
        ElMessage.warning('请选择一条记录');
        return;
    }
    if (operateType.value !== 'deleteProductAttr') {
        ElMessage.warning('请选择批量操作类型');
        return;
    }
    await handleDeleteProductAttr(multipleSelection.value.map(item => item.id));
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
// 处理更新操作
const handleUpdate = (index, row) => {
    router.push({ path: '/pms/updateProductAttr', query: { id: row.id } });
};
// 处理删除产品属性
const handleDeleteProductAttr = async (ids) => {
    try {
        await ElMessageBox.confirm('是否要删除该属性', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await productAttributeDeleteByIds({ ids: ids.join(",") });
        ElMessage.success('删除成功');
        getList();
    }
    catch (error) {
        console.error(error);
    }
};
// 处理删除操作
const handleDelete = async (index, row) => {
    await handleDeleteProductAttr([row.id]);
};
// 属性值的录入方式
const inputTypeFilter = (value) => {
    if (value === 1) {
        return '从列表中选取';
    }
    else {
        return '手工录入';
    }
};
// 属性是否可选
const selectTypeFilter = (value) => {
    if (value === 1) {
        return '单选';
    }
    else if (value === 2) {
        return '多选';
    }
    else {
        return '唯一';
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
    { onClick: (...[$event]) => {
            __VLS_ctx.addProductAttr();
            // @ts-ignore
            [addProductAttr,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_24 } = __VLS_20.slots;
// @ts-ignore
[];
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
    ref: "productAttrTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_27 = __VLS_26({
    ...{ 'onSelectionChange': {} },
    ref: "productAttrTable",
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
    width: "100",
    align: "center",
}));
const __VLS_42 = __VLS_41({
    label: "编号",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_41));
const { default: __VLS_45 } = __VLS_43.slots;
{
    const { default: __VLS_46 } = __VLS_43.slots;
    const [scope] = __VLS_vSlot(__VLS_46);
    (scope.row.id);
    // @ts-ignore
    [list, handleSelectionChange, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_43;
let __VLS_47;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({
    label: "属性名称",
    width: "140",
    align: "center",
}));
const __VLS_49 = __VLS_48({
    label: "属性名称",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_48));
const { default: __VLS_52 } = __VLS_50.slots;
{
    const { default: __VLS_53 } = __VLS_50.slots;
    const [scope] = __VLS_vSlot(__VLS_53);
    (scope.row.name);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_50;
let __VLS_54;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    label: "商品类型",
    width: "140",
    align: "center",
}));
const __VLS_56 = __VLS_55({
    label: "商品类型",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
const { default: __VLS_59 } = __VLS_57.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.template, __VLS_intrinsics.template)({});
(__VLS_ctx.$route.query.cname);
// @ts-ignore
[$route,];
var __VLS_57;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    label: "属性是否可选",
    width: "120",
    align: "center",
}));
const __VLS_62 = __VLS_61({
    label: "属性是否可选",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
const { default: __VLS_65 } = __VLS_63.slots;
{
    const { default: __VLS_66 } = __VLS_63.slots;
    const [scope] = __VLS_vSlot(__VLS_66);
    (__VLS_ctx.selectTypeFilter(scope.row.selectType));
    // @ts-ignore
    [selectTypeFilter,];
}
// @ts-ignore
[];
var __VLS_63;
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    label: "属性值的录入方式",
    width: "150",
    align: "center",
}));
const __VLS_69 = __VLS_68({
    label: "属性值的录入方式",
    width: "150",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
const { default: __VLS_72 } = __VLS_70.slots;
{
    const { default: __VLS_73 } = __VLS_70.slots;
    const [scope] = __VLS_vSlot(__VLS_73);
    (__VLS_ctx.inputTypeFilter(scope.row.inputType));
    // @ts-ignore
    [inputTypeFilter,];
}
// @ts-ignore
[];
var __VLS_70;
let __VLS_74;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_75 = __VLS_asFunctionalComponent1(__VLS_74, new __VLS_74({
    label: "可选值列表",
    align: "center",
}));
const __VLS_76 = __VLS_75({
    label: "可选值列表",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_75));
const { default: __VLS_79 } = __VLS_77.slots;
{
    const { default: __VLS_80 } = __VLS_77.slots;
    const [scope] = __VLS_vSlot(__VLS_80);
    (scope.row.inputList);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_77;
let __VLS_81;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_82 = __VLS_asFunctionalComponent1(__VLS_81, new __VLS_81({
    label: "排序",
    width: "100",
    align: "center",
}));
const __VLS_83 = __VLS_82({
    label: "排序",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_82));
const { default: __VLS_86 } = __VLS_84.slots;
{
    const { default: __VLS_87 } = __VLS_84.slots;
    const [scope] = __VLS_vSlot(__VLS_87);
    (scope.row.sort);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_84;
let __VLS_88;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_89 = __VLS_asFunctionalComponent1(__VLS_88, new __VLS_88({
    label: "操作",
    width: "200",
    align: "center",
}));
const __VLS_90 = __VLS_89({
    label: "操作",
    width: "200",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_89));
const { default: __VLS_93 } = __VLS_91.slots;
{
    const { default: __VLS_94 } = __VLS_91.slots;
    const [scope] = __VLS_vSlot(__VLS_94);
    let __VLS_95;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_96 = __VLS_asFunctionalComponent1(__VLS_95, new __VLS_95({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_97 = __VLS_96({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_96));
    let __VLS_100;
    const __VLS_101 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdate(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdate,];
            } });
    const { default: __VLS_102 } = __VLS_98.slots;
    // @ts-ignore
    [];
    var __VLS_98;
    var __VLS_99;
    let __VLS_103;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_104 = __VLS_asFunctionalComponent1(__VLS_103, new __VLS_103({
        ...{ 'onClick': {} },
        size: "small",
        type: "danger",
    }));
    const __VLS_105 = __VLS_104({
        ...{ 'onClick': {} },
        size: "small",
        type: "danger",
    }, ...__VLS_functionalComponentArgsRest(__VLS_104));
    let __VLS_108;
    const __VLS_109 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_110 } = __VLS_106.slots;
    // @ts-ignore
    [];
    var __VLS_106;
    var __VLS_107;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_91;
// @ts-ignore
[];
var __VLS_28;
var __VLS_29;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "batch-operate-container" },
});
/** @type {__VLS_StyleScopedClasses['batch-operate-container']} */ ;
let __VLS_111;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_112 = __VLS_asFunctionalComponent1(__VLS_111, new __VLS_111({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}));
const __VLS_113 = __VLS_112({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}, ...__VLS_functionalComponentArgsRest(__VLS_112));
const { default: __VLS_116 } = __VLS_114.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.operates))) {
    let __VLS_117;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_118 = __VLS_asFunctionalComponent1(__VLS_117, new __VLS_117({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_119 = __VLS_118({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_118));
    // @ts-ignore
    [operateType, operates,];
}
// @ts-ignore
[];
var __VLS_114;
let __VLS_122;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_123 = __VLS_asFunctionalComponent1(__VLS_122, new __VLS_122({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}));
const __VLS_124 = __VLS_123({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_123));
let __VLS_127;
const __VLS_128 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleBatchOperate();
            // @ts-ignore
            [handleBatchOperate,];
        } });
/** @type {__VLS_StyleScopedClasses['search-button']} */ ;
const { default: __VLS_129 } = __VLS_125.slots;
// @ts-ignore
[];
var __VLS_125;
var __VLS_126;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_130;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_131 = __VLS_asFunctionalComponent1(__VLS_130, new __VLS_130({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    currentPage: (__VLS_ctx.listQuery.pageNum),
    total: (__VLS_ctx.total),
}));
const __VLS_132 = __VLS_131({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    currentPage: (__VLS_ctx.listQuery.pageNum),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_131));
let __VLS_135;
const __VLS_136 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_137 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_133;
var __VLS_134;
// @ts-ignore
var __VLS_33 = __VLS_32;
// @ts-ignore
[listQuery, listQuery, total, handleSizeChange, handleCurrentChange,];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
