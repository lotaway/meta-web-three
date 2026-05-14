/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRoute } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getFlashProductRelationListAPI, flashProductRelationCreateAPI, flashProductRelationDeleteByIdAPI, flashProductRelationUpdateByIdAPI } from '@/apis/flashProductRelation';
import { getProductListAPI } from '@/apis/product';
// 获取路由
const route = useRoute();
// 秒杀商品关系列表数据
const listQuery = ref({
    pageNum: 1,
    pageSize: 10
});
// 秒杀商品列数据
const list = ref([]);
// 总条数
const total = ref(0);
// 加载状态
const listLoading = ref(false);
// 获取列表数据
const getList = async () => {
    listLoading.value = true;
    try {
        const res = await getFlashProductRelationListAPI(listQuery.value);
        listLoading.value = false;
        list.value = res.data.list;
        total.value = res.data.total;
    }
    catch (error) {
        listLoading.value = false;
        console.error('获取秒杀商品列表失败:', error);
    }
};
// 组件挂载时获取数据
onMounted(() => {
    listQuery.value.flashPromotionId = Number(route.query.flashPromotionId);
    listQuery.value.flashPromotionSessionId = Number(route.query.flashPromotionSessionId);
    getList();
});
// 秒杀商品关系编辑框可见性
const editDialogVisible = ref(false);
// 当前操作的秒杀商品关系
const flashProductRelation = ref();
// 对话框数据
const dialogData = ref({
    list: [],
    total: 0,
    multipleSelection: [],
    listQuery: {
        keyword: '',
        pageNum: 1,
        pageSize: 5
    }
});
// 选择商品对话框可见性
const selectDialogVisible = ref(false);
// 每页大小改变
const handleSizeChange = (val) => {
    listQuery.value.pageNum = 1;
    listQuery.value.pageSize = val;
    getList();
};
// 当前页改变
const handleCurrentChange = (val) => {
    listQuery.value.pageNum = val;
    getList();
};
// 选择商品
const handleSelectProduct = () => {
    selectDialogVisible.value = true;
    getDialogList();
};
// 更新秒杀商品
const handleUpdate = (index, row) => {
    editDialogVisible.value = true;
    flashProductRelation.value = Object.assign({}, row);
};
// 删除秒杀商品
const handleDelete = async (index, row) => {
    try {
        await ElMessageBox.confirm('是否要删除该商品?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await flashProductRelationDeleteByIdAPI(row.id);
        ElMessage.success('删除成功!');
        getList();
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('删除秒杀商品失败:', error);
        }
    }
};
// 选择搜索
const handleSelectSearch = () => {
    getDialogList();
};
// 对话框每页大小改变
const handleDialogSizeChange = (val) => {
    dialogData.value.listQuery.pageNum = 1;
    dialogData.value.listQuery.pageSize = val;
    getDialogList();
};
// 对话框当前页改变
const handleDialogCurrentChange = (val) => {
    dialogData.value.listQuery.pageNum = val;
    getDialogList();
};
// 对话框选中项改变
const handleDialogSelectionChange = (val) => {
    dialogData.value.multipleSelection = val;
};
// 选择对话框确认
const handleSelectDialogConfirm = async () => {
    if (dialogData.value.multipleSelection.length < 1) {
        ElMessage.warning('请选择一条记录');
        return;
    }
    const selectProducts = dialogData.value.multipleSelection.map(item => {
        return {
            productId: item.id,
            flashPromotionId: listQuery.value.flashPromotionId,
            flashPromotionSessionId: listQuery.value.flashPromotionSessionId
        };
    });
    try {
        await ElMessageBox.confirm('使用要进行添加操作?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await flashProductRelationCreateAPI(selectProducts);
        selectDialogVisible.value = false;
        dialogData.value.multipleSelection = [];
        getList();
        ElMessage.success('添加成功!');
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('添加秒杀商品失败:', error);
        }
    }
};
// 编辑对话框确认
const handleEditDialogConfirm = async () => {
    try {
        await ElMessageBox.confirm('是否要确认?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await flashProductRelationUpdateByIdAPI(flashProductRelation.value.id, flashProductRelation.value);
        ElMessage.success('修改成功！');
        editDialogVisible.value = false;
        getList();
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('更新秒杀商品失败:', error);
        }
    }
};
// 获取对话框商品列表数据
const getDialogList = async () => {
    try {
        const res = await getProductListAPI(dialogData.value.listQuery);
        dialogData.value.list = res.data.list;
        dialogData.value.total = res.data.total;
    }
    catch (error) {
        console.error('获取商品列表失败:', error);
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
__VLS_asFunctionalElement1(__VLS_intrinsics.i, __VLS_intrinsics.i)({
    ...{ class: "el-icon-tickets" },
});
/** @type {__VLS_StyleScopedClasses['el-icon-tickets']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
let __VLS_6;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
    ...{ style: {} },
}));
const __VLS_8 = __VLS_7({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
let __VLS_11;
const __VLS_12 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleSelectProduct();
            // @ts-ignore
            [handleSelectProduct,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_13 } = __VLS_9.slots;
// @ts-ignore
[];
var __VLS_9;
var __VLS_10;
// @ts-ignore
[];
var __VLS_3;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_14;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_15 = __VLS_asFunctionalComponent1(__VLS_14, new __VLS_14({
    ref: "productRelationTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_16 = __VLS_15({
    ref: "productRelationTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_19 = {};
const { default: __VLS_21 } = __VLS_17.slots;
let __VLS_22;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_23 = __VLS_asFunctionalComponent1(__VLS_22, new __VLS_22({
    label: "编号",
    width: "100",
    align: "center",
}));
const __VLS_24 = __VLS_23({
    label: "编号",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_23));
const { default: __VLS_27 } = __VLS_25.slots;
{
    const { default: __VLS_28 } = __VLS_25.slots;
    const [scope] = __VLS_vSlot(__VLS_28);
    (scope.row.id);
    // @ts-ignore
    [list, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_25;
let __VLS_29;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_30 = __VLS_asFunctionalComponent1(__VLS_29, new __VLS_29({
    label: "商品名称",
    align: "center",
}));
const __VLS_31 = __VLS_30({
    label: "商品名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_30));
const { default: __VLS_34 } = __VLS_32.slots;
{
    const { default: __VLS_35 } = __VLS_32.slots;
    const [scope] = __VLS_vSlot(__VLS_35);
    (scope.row.product.name);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_32;
let __VLS_36;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_37 = __VLS_asFunctionalComponent1(__VLS_36, new __VLS_36({
    label: "货号",
    width: "140",
    align: "center",
}));
const __VLS_38 = __VLS_37({
    label: "货号",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_37));
const { default: __VLS_41 } = __VLS_39.slots;
{
    const { default: __VLS_42 } = __VLS_39.slots;
    const [scope] = __VLS_vSlot(__VLS_42);
    (scope.row.product.productSn);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_39;
let __VLS_43;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_44 = __VLS_asFunctionalComponent1(__VLS_43, new __VLS_43({
    label: "商品价格",
    width: "100",
    align: "center",
}));
const __VLS_45 = __VLS_44({
    label: "商品价格",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_44));
const { default: __VLS_48 } = __VLS_46.slots;
{
    const { default: __VLS_49 } = __VLS_46.slots;
    const [scope] = __VLS_vSlot(__VLS_49);
    (scope.row.product.price);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_46;
let __VLS_50;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_51 = __VLS_asFunctionalComponent1(__VLS_50, new __VLS_50({
    label: "剩余数量",
    width: "100",
    align: "center",
}));
const __VLS_52 = __VLS_51({
    label: "剩余数量",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_51));
const { default: __VLS_55 } = __VLS_53.slots;
{
    const { default: __VLS_56 } = __VLS_53.slots;
    const [scope] = __VLS_vSlot(__VLS_56);
    (scope.row.product.stock);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_53;
let __VLS_57;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_58 = __VLS_asFunctionalComponent1(__VLS_57, new __VLS_57({
    label: "秒杀价格",
    width: "100",
    align: "center",
}));
const __VLS_59 = __VLS_58({
    label: "秒杀价格",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_58));
const { default: __VLS_62 } = __VLS_60.slots;
{
    const { default: __VLS_63 } = __VLS_60.slots;
    const [scope] = __VLS_vSlot(__VLS_63);
    if (scope.row.flashPromotionPrice !== null) {
        __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
        (scope.row.flashPromotionPrice);
    }
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_60;
let __VLS_64;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_65 = __VLS_asFunctionalComponent1(__VLS_64, new __VLS_64({
    label: "秒杀数量",
    width: "100",
    align: "center",
}));
const __VLS_66 = __VLS_65({
    label: "秒杀数量",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_65));
const { default: __VLS_69 } = __VLS_67.slots;
{
    const { default: __VLS_70 } = __VLS_67.slots;
    const [scope] = __VLS_vSlot(__VLS_70);
    (scope.row.flashPromotionCount);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_67;
let __VLS_71;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_72 = __VLS_asFunctionalComponent1(__VLS_71, new __VLS_71({
    label: "限购数量",
    width: "100",
    align: "center",
}));
const __VLS_73 = __VLS_72({
    label: "限购数量",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_72));
const { default: __VLS_76 } = __VLS_74.slots;
{
    const { default: __VLS_77 } = __VLS_74.slots;
    const [scope] = __VLS_vSlot(__VLS_77);
    (scope.row.flashPromotionLimit);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_74;
let __VLS_78;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_79 = __VLS_asFunctionalComponent1(__VLS_78, new __VLS_78({
    label: "排序",
    width: "100",
    align: "center",
}));
const __VLS_80 = __VLS_79({
    label: "排序",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_79));
const { default: __VLS_83 } = __VLS_81.slots;
{
    const { default: __VLS_84 } = __VLS_81.slots;
    const [scope] = __VLS_vSlot(__VLS_84);
    (scope.row.sort);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_81;
let __VLS_85;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_86 = __VLS_asFunctionalComponent1(__VLS_85, new __VLS_85({
    label: "操作",
    width: "100",
    align: "center",
}));
const __VLS_87 = __VLS_86({
    label: "操作",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_86));
const { default: __VLS_90 } = __VLS_88.slots;
{
    const { default: __VLS_91 } = __VLS_88.slots;
    const [scope] = __VLS_vSlot(__VLS_91);
    let __VLS_92;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_93 = __VLS_asFunctionalComponent1(__VLS_92, new __VLS_92({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_94 = __VLS_93({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_93));
    let __VLS_97;
    const __VLS_98 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdate(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdate,];
            } });
    const { default: __VLS_99 } = __VLS_95.slots;
    // @ts-ignore
    [];
    var __VLS_95;
    var __VLS_96;
    let __VLS_100;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_101 = __VLS_asFunctionalComponent1(__VLS_100, new __VLS_100({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_102 = __VLS_101({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_101));
    let __VLS_105;
    const __VLS_106 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_107 } = __VLS_103.slots;
    // @ts-ignore
    [];
    var __VLS_103;
    var __VLS_104;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_88;
// @ts-ignore
[];
var __VLS_17;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_108;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_109 = __VLS_asFunctionalComponent1(__VLS_108, new __VLS_108({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}));
const __VLS_110 = __VLS_109({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_109));
let __VLS_113;
const __VLS_114 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_115 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_111;
var __VLS_112;
let __VLS_116;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_117 = __VLS_asFunctionalComponent1(__VLS_116, new __VLS_116({
    title: "选择商品",
    modelValue: (__VLS_ctx.selectDialogVisible),
    width: "50%",
}));
const __VLS_118 = __VLS_117({
    title: "选择商品",
    modelValue: (__VLS_ctx.selectDialogVisible),
    width: "50%",
}, ...__VLS_functionalComponentArgsRest(__VLS_117));
const { default: __VLS_121 } = __VLS_119.slots;
let __VLS_122;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_123 = __VLS_asFunctionalComponent1(__VLS_122, new __VLS_122({
    modelValue: (__VLS_ctx.dialogData.listQuery.keyword),
    ...{ style: {} },
    size: "small",
    placeholder: "商品名称搜索",
}));
const __VLS_124 = __VLS_123({
    modelValue: (__VLS_ctx.dialogData.listQuery.keyword),
    ...{ style: {} },
    size: "small",
    placeholder: "商品名称搜索",
}, ...__VLS_functionalComponentArgsRest(__VLS_123));
const { default: __VLS_127 } = __VLS_125.slots;
{
    const { append: __VLS_128 } = __VLS_125.slots;
    let __VLS_129;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_130 = __VLS_asFunctionalComponent1(__VLS_129, new __VLS_129({
        ...{ 'onClick': {} },
        icon: "el-icon-search",
    }));
    const __VLS_131 = __VLS_130({
        ...{ 'onClick': {} },
        icon: "el-icon-search",
    }, ...__VLS_functionalComponentArgsRest(__VLS_130));
    let __VLS_134;
    const __VLS_135 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleSelectSearch();
                // @ts-ignore
                [listQuery, listQuery, total, handleSizeChange, handleCurrentChange, selectDialogVisible, dialogData, handleSelectSearch,];
            } });
    var __VLS_132;
    var __VLS_133;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_125;
let __VLS_136;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_137 = __VLS_asFunctionalComponent1(__VLS_136, new __VLS_136({
    ...{ 'onSelectionChange': {} },
    data: (__VLS_ctx.dialogData.list),
    border: true,
}));
const __VLS_138 = __VLS_137({
    ...{ 'onSelectionChange': {} },
    data: (__VLS_ctx.dialogData.list),
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_137));
let __VLS_141;
const __VLS_142 = ({ selectionChange: {} },
    { onSelectionChange: (__VLS_ctx.handleDialogSelectionChange) });
const { default: __VLS_143 } = __VLS_139.slots;
let __VLS_144;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_145 = __VLS_asFunctionalComponent1(__VLS_144, new __VLS_144({
    type: "selection",
    width: "60",
    align: "center",
}));
const __VLS_146 = __VLS_145({
    type: "selection",
    width: "60",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_145));
let __VLS_149;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_150 = __VLS_asFunctionalComponent1(__VLS_149, new __VLS_149({
    label: "商品名称",
    align: "center",
}));
const __VLS_151 = __VLS_150({
    label: "商品名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_150));
const { default: __VLS_154 } = __VLS_152.slots;
{
    const { default: __VLS_155 } = __VLS_152.slots;
    const [scope] = __VLS_vSlot(__VLS_155);
    (scope.row.name);
    // @ts-ignore
    [dialogData, handleDialogSelectionChange,];
}
// @ts-ignore
[];
var __VLS_152;
let __VLS_156;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_157 = __VLS_asFunctionalComponent1(__VLS_156, new __VLS_156({
    label: "货号",
    width: "160",
    align: "center",
}));
const __VLS_158 = __VLS_157({
    label: "货号",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_157));
const { default: __VLS_161 } = __VLS_159.slots;
{
    const { default: __VLS_162 } = __VLS_159.slots;
    const [scope] = __VLS_vSlot(__VLS_162);
    (scope.row.productSn);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_159;
let __VLS_163;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_164 = __VLS_asFunctionalComponent1(__VLS_163, new __VLS_163({
    label: "价格",
    width: "120",
    align: "center",
}));
const __VLS_165 = __VLS_164({
    label: "价格",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_164));
const { default: __VLS_168 } = __VLS_166.slots;
{
    const { default: __VLS_169 } = __VLS_166.slots;
    const [scope] = __VLS_vSlot(__VLS_169);
    (scope.row.price);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_166;
// @ts-ignore
[];
var __VLS_139;
var __VLS_140;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_170;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_171 = __VLS_asFunctionalComponent1(__VLS_170, new __VLS_170({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "prev, pager, next",
    currentPage: (__VLS_ctx.dialogData.listQuery.pageNum),
    pageSize: (__VLS_ctx.dialogData.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.dialogData.total),
}));
const __VLS_172 = __VLS_171({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "prev, pager, next",
    currentPage: (__VLS_ctx.dialogData.listQuery.pageNum),
    pageSize: (__VLS_ctx.dialogData.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.dialogData.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_171));
let __VLS_175;
const __VLS_176 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleDialogSizeChange) });
const __VLS_177 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleDialogCurrentChange) });
var __VLS_173;
var __VLS_174;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
{
    const { footer: __VLS_178 } = __VLS_119.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
    let __VLS_179;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_180 = __VLS_asFunctionalComponent1(__VLS_179, new __VLS_179({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_181 = __VLS_180({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_180));
    let __VLS_184;
    const __VLS_185 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.selectDialogVisible = false;
                // @ts-ignore
                [selectDialogVisible, dialogData, dialogData, dialogData, handleDialogSizeChange, handleDialogCurrentChange,];
            } });
    const { default: __VLS_186 } = __VLS_182.slots;
    // @ts-ignore
    [];
    var __VLS_182;
    var __VLS_183;
    let __VLS_187;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_188 = __VLS_asFunctionalComponent1(__VLS_187, new __VLS_187({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
    }));
    const __VLS_189 = __VLS_188({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_188));
    let __VLS_192;
    const __VLS_193 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleSelectDialogConfirm();
                // @ts-ignore
                [handleSelectDialogConfirm,];
            } });
    const { default: __VLS_194 } = __VLS_190.slots;
    // @ts-ignore
    [];
    var __VLS_190;
    var __VLS_191;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_119;
let __VLS_195;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_196 = __VLS_asFunctionalComponent1(__VLS_195, new __VLS_195({
    title: "编辑秒杀商品信息",
    modelValue: (__VLS_ctx.editDialogVisible),
    width: "40%",
}));
const __VLS_197 = __VLS_196({
    title: "编辑秒杀商品信息",
    modelValue: (__VLS_ctx.editDialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_196));
const { default: __VLS_200 } = __VLS_198.slots;
if (__VLS_ctx.flashProductRelation?.product) {
    let __VLS_201;
    /** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
    elForm;
    // @ts-ignore
    const __VLS_202 = __VLS_asFunctionalComponent1(__VLS_201, new __VLS_201({
        model: (__VLS_ctx.flashProductRelation),
        ref: "flashProductRelationForm",
        labelWidth: "150px",
    }));
    const __VLS_203 = __VLS_202({
        model: (__VLS_ctx.flashProductRelation),
        ref: "flashProductRelationForm",
        labelWidth: "150px",
    }, ...__VLS_functionalComponentArgsRest(__VLS_202));
    var __VLS_206 = {};
    const { default: __VLS_208 } = __VLS_204.slots;
    let __VLS_209;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_210 = __VLS_asFunctionalComponent1(__VLS_209, new __VLS_209({
        label: "商品名称：",
    }));
    const __VLS_211 = __VLS_210({
        label: "商品名称：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_210));
    const { default: __VLS_214 } = __VLS_212.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
    (__VLS_ctx.flashProductRelation.product.name);
    // @ts-ignore
    [editDialogVisible, flashProductRelation, flashProductRelation, flashProductRelation,];
    var __VLS_212;
    let __VLS_215;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_216 = __VLS_asFunctionalComponent1(__VLS_215, new __VLS_215({
        label: "货号：",
    }));
    const __VLS_217 = __VLS_216({
        label: "货号：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_216));
    const { default: __VLS_220 } = __VLS_218.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
    (__VLS_ctx.flashProductRelation.product.productSn);
    // @ts-ignore
    [flashProductRelation,];
    var __VLS_218;
    let __VLS_221;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_222 = __VLS_asFunctionalComponent1(__VLS_221, new __VLS_221({
        label: "商品价格：",
    }));
    const __VLS_223 = __VLS_222({
        label: "商品价格：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_222));
    const { default: __VLS_226 } = __VLS_224.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
    (__VLS_ctx.flashProductRelation.product.price);
    // @ts-ignore
    [flashProductRelation,];
    var __VLS_224;
    let __VLS_227;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_228 = __VLS_asFunctionalComponent1(__VLS_227, new __VLS_227({
        label: "秒杀价格：",
    }));
    const __VLS_229 = __VLS_228({
        label: "秒杀价格：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_228));
    const { default: __VLS_232 } = __VLS_230.slots;
    let __VLS_233;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_234 = __VLS_asFunctionalComponent1(__VLS_233, new __VLS_233({
        modelValue: (__VLS_ctx.flashProductRelation.flashPromotionPrice),
        ...{ class: "input-width" },
    }));
    const __VLS_235 = __VLS_234({
        modelValue: (__VLS_ctx.flashProductRelation.flashPromotionPrice),
        ...{ class: "input-width" },
    }, ...__VLS_functionalComponentArgsRest(__VLS_234));
    /** @type {__VLS_StyleScopedClasses['input-width']} */ ;
    const { default: __VLS_238 } = __VLS_236.slots;
    {
        const { prepend: __VLS_239 } = __VLS_236.slots;
        // @ts-ignore
        [flashProductRelation,];
    }
    // @ts-ignore
    [];
    var __VLS_236;
    // @ts-ignore
    [];
    var __VLS_230;
    let __VLS_240;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_241 = __VLS_asFunctionalComponent1(__VLS_240, new __VLS_240({
        label: "剩余数量：",
    }));
    const __VLS_242 = __VLS_241({
        label: "剩余数量：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_241));
    const { default: __VLS_245 } = __VLS_243.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
    (__VLS_ctx.flashProductRelation.product.stock);
    // @ts-ignore
    [flashProductRelation,];
    var __VLS_243;
    let __VLS_246;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_247 = __VLS_asFunctionalComponent1(__VLS_246, new __VLS_246({
        label: "秒杀数量：",
    }));
    const __VLS_248 = __VLS_247({
        label: "秒杀数量：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_247));
    const { default: __VLS_251 } = __VLS_249.slots;
    let __VLS_252;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_253 = __VLS_asFunctionalComponent1(__VLS_252, new __VLS_252({
        modelValue: (__VLS_ctx.flashProductRelation.flashPromotionCount),
        ...{ class: "input-width" },
    }));
    const __VLS_254 = __VLS_253({
        modelValue: (__VLS_ctx.flashProductRelation.flashPromotionCount),
        ...{ class: "input-width" },
    }, ...__VLS_functionalComponentArgsRest(__VLS_253));
    /** @type {__VLS_StyleScopedClasses['input-width']} */ ;
    // @ts-ignore
    [flashProductRelation,];
    var __VLS_249;
    let __VLS_257;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_258 = __VLS_asFunctionalComponent1(__VLS_257, new __VLS_257({
        label: "限购数量：",
    }));
    const __VLS_259 = __VLS_258({
        label: "限购数量：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_258));
    const { default: __VLS_262 } = __VLS_260.slots;
    let __VLS_263;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_264 = __VLS_asFunctionalComponent1(__VLS_263, new __VLS_263({
        modelValue: (__VLS_ctx.flashProductRelation.flashPromotionLimit),
        ...{ class: "input-width" },
    }));
    const __VLS_265 = __VLS_264({
        modelValue: (__VLS_ctx.flashProductRelation.flashPromotionLimit),
        ...{ class: "input-width" },
    }, ...__VLS_functionalComponentArgsRest(__VLS_264));
    /** @type {__VLS_StyleScopedClasses['input-width']} */ ;
    // @ts-ignore
    [flashProductRelation,];
    var __VLS_260;
    let __VLS_268;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_269 = __VLS_asFunctionalComponent1(__VLS_268, new __VLS_268({
        label: "排序：",
    }));
    const __VLS_270 = __VLS_269({
        label: "排序：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_269));
    const { default: __VLS_273 } = __VLS_271.slots;
    let __VLS_274;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_275 = __VLS_asFunctionalComponent1(__VLS_274, new __VLS_274({
        modelValue: (__VLS_ctx.flashProductRelation.sort),
        ...{ class: "input-width" },
    }));
    const __VLS_276 = __VLS_275({
        modelValue: (__VLS_ctx.flashProductRelation.sort),
        ...{ class: "input-width" },
    }, ...__VLS_functionalComponentArgsRest(__VLS_275));
    /** @type {__VLS_StyleScopedClasses['input-width']} */ ;
    // @ts-ignore
    [flashProductRelation,];
    var __VLS_271;
    // @ts-ignore
    [];
    var __VLS_204;
}
{
    const { footer: __VLS_279 } = __VLS_198.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_280;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_281 = __VLS_asFunctionalComponent1(__VLS_280, new __VLS_280({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_282 = __VLS_281({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_281));
    let __VLS_285;
    const __VLS_286 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.editDialogVisible = false;
                // @ts-ignore
                [editDialogVisible,];
            } });
    const { default: __VLS_287 } = __VLS_283.slots;
    // @ts-ignore
    [];
    var __VLS_283;
    var __VLS_284;
    let __VLS_288;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_289 = __VLS_asFunctionalComponent1(__VLS_288, new __VLS_288({
        ...{ 'onClick': {} },
        type: "primary",
        size: "small",
    }));
    const __VLS_290 = __VLS_289({
        ...{ 'onClick': {} },
        type: "primary",
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_289));
    let __VLS_293;
    const __VLS_294 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleEditDialogConfirm();
                // @ts-ignore
                [handleEditDialogConfirm,];
            } });
    const { default: __VLS_295 } = __VLS_291.slots;
    // @ts-ignore
    [];
    var __VLS_291;
    var __VLS_292;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_198;
// @ts-ignore
var __VLS_20 = __VLS_19, __VLS_207 = __VLS_206;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
