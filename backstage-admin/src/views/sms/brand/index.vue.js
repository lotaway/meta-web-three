/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { Search, Tickets } from '@element-plus/icons-vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getHomeBrandListAPI, homeBrandUpdateRecommendStatusAPI, homeBrandDeleteByIdsAPI, homeBrandCreateAPI, homeBrandUpdateSortAPI } from '@/apis/homeBrand';
import { getBrandListAPI } from '@/apis/brand';
// 列表查询参数
const listQuery = ref({
    pageNum: 1,
    pageSize: 10,
});
// 首页品牌列表数据
const list = ref([]);
// 总条数
const total = ref(0);
// 加载状态
const listLoading = ref(false);
// 获取列表数据
const getList = async () => {
    listLoading.value = true;
    try {
        const res = await getHomeBrandListAPI(listQuery.value);
        listLoading.value = false;
        list.value = res.data.list;
        total.value = res.data.total;
    }
    catch (error) {
        listLoading.value = false;
        console.error('获取列表失败:', error);
    }
};
// 页面加载完成后获取数据
onMounted(() => {
    getList();
});
// 表格中被选中的行
const multipleSelection = ref([]);
const recommendOptions = ref([
    {
        label: '未推荐',
        value: 0
    },
    {
        label: '推荐中',
        value: 1
    }
]);
const operates = ref([
    {
        label: "设为推荐",
        value: 0
    },
    {
        label: "取消推荐",
        value: 1
    },
    {
        label: "删除",
        value: 2
    }
]);
const operateType = ref();
// 选择品牌对话框可见性
const selectDialogVisible = ref(false);
// 选择品牌对话框中的数据
const dialogData = ref({
    list: [],
    total: 0,
    multipleSelection: [],
    listQuery: {
        keyword: '',
        showStatus: 1,
        pageNum: 1,
        pageSize: 5
    }
});
// 获取选择品牌对话框列表
const getDialogList = async () => {
    try {
        const res = await getBrandListAPI(dialogData.value.listQuery);
        dialogData.value.list = res.data.list;
        dialogData.value.total = res.data.total;
    }
    catch (error) {
        console.error('获取对话框列表失败:', error);
    }
};
// 设置排序对话框可见性
const sortDialogVisible = ref(false);
// 设置排序对话框中的数据
const sortDialogData = ref({ sort: 0, id: 0 });
// 重置搜索
const handleResetSearch = () => {
    listQuery.value = {
        pageNum: 1,
        pageSize: 10,
    };
};
// 搜索列表
const handleSearchList = async () => {
    listQuery.value.pageNum = 1;
    await getList();
};
// 处理表格选中变化
const handleSelectionChange = (val) => {
    multipleSelection.value = val;
};
// 每页大小变化
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
// 处理推荐状态变化
const handleRecommendStatusStatusChange = async (index, row) => {
    await updateRecommendStatusStatus([row.id], row.recommendStatus);
};
// 处理删除
const handleDelete = async (index, row) => {
    await deleteBrand([row.id]);
};
// 处理批量操作
const handleBatchOperate = async () => {
    if (multipleSelection.value.length < 1) {
        ElMessage.warning({
            message: '请选择一条记录',
            duration: 1000
        });
        return;
    }
    const ids = multipleSelection.value.map(item => item.id);
    if (operateType.value === 0) {
        // 设为推荐
        await updateRecommendStatusStatus(ids, 1);
    }
    else if (operateType.value === 1) {
        // 取消推荐
        await updateRecommendStatusStatus(ids, 0);
    }
    else if (operateType.value === 2) {
        // 删除
        await deleteBrand(ids);
    }
    else {
        ElMessage.warning({
            message: '请选择批量操作类型',
            duration: 1000
        });
        return;
    }
};
// 处理选择品牌
const handleSelectBrand = () => {
    selectDialogVisible.value = true;
    dialogData.value.listQuery.keyword = '';
    getDialogList();
};
// 处理对话框搜索
const handleSelectSearch = () => {
    getDialogList();
};
// 对话框每页大小变化
const handleDialogSizeChange = (val) => {
    dialogData.value.listQuery.pageNum = 1;
    dialogData.value.listQuery.pageSize = val;
    getDialogList();
};
// 对话框当前页变化
const handleDialogCurrentChange = (val) => {
    dialogData.value.listQuery.pageNum = val;
    getDialogList();
};
// 对话框选中变化
const handleDialogSelectionChange = (val) => {
    dialogData.value.multipleSelection = val;
};
// 确认选择对话框
const handleSelectDialogConfirm = async () => {
    if (dialogData.value.multipleSelection.length < 1) {
        ElMessage.warning({
            message: '请选择一条记录',
            duration: 1000
        });
        return;
    }
    try {
        await ElMessageBox.confirm('使用要进行添加操作?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        const homeBrandList = dialogData.value.multipleSelection.map(item => ({
            brandId: item.id,
            brandName: item.name
        }));
        await homeBrandCreateAPI(homeBrandList);
        selectDialogVisible.value = false;
        dialogData.value.multipleSelection = [];
        await getList();
        ElMessage.success({
            type: 'success',
            message: '添加成功!'
        });
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('确认操作失败:', error);
        }
    }
};
// 编辑排序
const handleEditSort = (index, row) => {
    sortDialogVisible.value = true;
    sortDialogData.value.sort = row.sort;
    sortDialogData.value.id = row.id;
};
// 更新排序
const handleUpdateSort = async () => {
    if (!sortDialogData.value.id)
        return;
    try {
        await ElMessageBox.confirm('是否要修改排序?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await homeBrandUpdateSortAPI(sortDialogData.value);
        sortDialogVisible.value = false;
        await getList();
        ElMessage.success({
            type: 'success',
            message: '修改成功!'
        });
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('修改排序失败:', error);
        }
    }
};
// 更新推荐状态
const updateRecommendStatusStatus = async (ids, status) => {
    try {
        await ElMessageBox.confirm('是否要修改推荐状态?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await homeBrandUpdateRecommendStatusAPI({
            ids: ids.join(','),
            recommendStatus: status
        });
        await getList();
        ElMessage.success({
            type: 'success',
            message: '修改成功!'
        });
    }
    catch (error) {
        console.error('修改失败:', error);
        await getList();
    }
};
// 删除品牌
const deleteBrand = async (ids) => {
    try {
        await ElMessageBox.confirm('是否要删除该推荐?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await homeBrandDeleteByIdsAPI({ ids: ids.join(',') });
        await getList();
        ElMessage.success({
            type: 'success',
            message: '删除成功!'
        });
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('删除操作失败:', error);
        }
    }
};
// 推荐状态过滤器
const formatRecommendStatus = (status) => {
    if (status === 1) {
        return '推荐中';
    }
    else {
        return '未推荐';
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
    label: "品牌名称：",
}));
const __VLS_41 = __VLS_40({
    label: "品牌名称：",
}, ...__VLS_functionalComponentArgsRest(__VLS_40));
const { default: __VLS_44 } = __VLS_42.slots;
let __VLS_45;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_46 = __VLS_asFunctionalComponent1(__VLS_45, new __VLS_45({
    modelValue: (__VLS_ctx.listQuery.brandName),
    placeholder: "品牌名称",
}));
const __VLS_47 = __VLS_46({
    modelValue: (__VLS_ctx.listQuery.brandName),
    placeholder: "品牌名称",
}, ...__VLS_functionalComponentArgsRest(__VLS_46));
// @ts-ignore
[listQuery, listQuery,];
var __VLS_42;
let __VLS_50;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_51 = __VLS_asFunctionalComponent1(__VLS_50, new __VLS_50({
    label: "推荐状态：",
}));
const __VLS_52 = __VLS_51({
    label: "推荐状态：",
}, ...__VLS_functionalComponentArgsRest(__VLS_51));
const { default: __VLS_55 } = __VLS_53.slots;
let __VLS_56;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
    modelValue: (__VLS_ctx.listQuery.recommendStatus),
    placeholder: "全部",
    clearable: true,
    ...{ style: {} },
}));
const __VLS_58 = __VLS_57({
    modelValue: (__VLS_ctx.listQuery.recommendStatus),
    placeholder: "全部",
    clearable: true,
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_57));
const { default: __VLS_61 } = __VLS_59.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.recommendOptions))) {
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
    [listQuery, recommendOptions,];
}
// @ts-ignore
[];
var __VLS_59;
// @ts-ignore
[];
var __VLS_53;
// @ts-ignore
[];
var __VLS_36;
// @ts-ignore
[];
var __VLS_3;
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    ...{ class: "operate-container" },
    shadow: "never",
}));
const __VLS_69 = __VLS_68({
    ...{ class: "operate-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
const { default: __VLS_72 } = __VLS_70.slots;
let __VLS_73;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_74 = __VLS_asFunctionalComponent1(__VLS_73, new __VLS_73({
    ...{ class: "el-icon-middle" },
}));
const __VLS_75 = __VLS_74({
    ...{ class: "el-icon-middle" },
}, ...__VLS_functionalComponentArgsRest(__VLS_74));
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_78 } = __VLS_76.slots;
let __VLS_79;
/** @ts-ignore @type { | typeof __VLS_components.Tickets} */
Tickets;
// @ts-ignore
const __VLS_80 = __VLS_asFunctionalComponent1(__VLS_79, new __VLS_79({}));
const __VLS_81 = __VLS_80({}, ...__VLS_functionalComponentArgsRest(__VLS_80));
// @ts-ignore
[];
var __VLS_76;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
let __VLS_84;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_85 = __VLS_asFunctionalComponent1(__VLS_84, new __VLS_84({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}));
const __VLS_86 = __VLS_85({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}, ...__VLS_functionalComponentArgsRest(__VLS_85));
let __VLS_89;
const __VLS_90 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleSelectBrand();
            // @ts-ignore
            [handleSelectBrand,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_91 } = __VLS_87.slots;
// @ts-ignore
[];
var __VLS_87;
var __VLS_88;
// @ts-ignore
[];
var __VLS_70;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_92;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_93 = __VLS_asFunctionalComponent1(__VLS_92, new __VLS_92({
    ...{ 'onSelectionChange': {} },
    ref: "homeBrandTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_94 = __VLS_93({
    ...{ 'onSelectionChange': {} },
    ref: "homeBrandTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_93));
let __VLS_97;
const __VLS_98 = ({ selectionChange: {} },
    { onSelectionChange: (__VLS_ctx.handleSelectionChange) });
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_99 = {};
const { default: __VLS_101 } = __VLS_95.slots;
let __VLS_102;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_103 = __VLS_asFunctionalComponent1(__VLS_102, new __VLS_102({
    type: "selection",
    width: "60",
    align: "center",
}));
const __VLS_104 = __VLS_103({
    type: "selection",
    width: "60",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_103));
let __VLS_107;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_108 = __VLS_asFunctionalComponent1(__VLS_107, new __VLS_107({
    label: "编号",
    width: "120",
    align: "center",
}));
const __VLS_109 = __VLS_108({
    label: "编号",
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_108));
const { default: __VLS_112 } = __VLS_110.slots;
{
    const { default: __VLS_113 } = __VLS_110.slots;
    const [scope] = __VLS_vSlot(__VLS_113);
    (scope.row.id);
    // @ts-ignore
    [list, handleSelectionChange, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_110;
let __VLS_114;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_115 = __VLS_asFunctionalComponent1(__VLS_114, new __VLS_114({
    label: "品牌名称",
    align: "center",
}));
const __VLS_116 = __VLS_115({
    label: "品牌名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_115));
const { default: __VLS_119 } = __VLS_117.slots;
{
    const { default: __VLS_120 } = __VLS_117.slots;
    const [scope] = __VLS_vSlot(__VLS_120);
    (scope.row.brandName);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_117;
let __VLS_121;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_122 = __VLS_asFunctionalComponent1(__VLS_121, new __VLS_121({
    label: "是否推荐",
    width: "200",
    align: "center",
}));
const __VLS_123 = __VLS_122({
    label: "是否推荐",
    width: "200",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_122));
const { default: __VLS_126 } = __VLS_124.slots;
{
    const { default: __VLS_127 } = __VLS_124.slots;
    const [scope] = __VLS_vSlot(__VLS_127);
    let __VLS_128;
    /** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
    elSwitch;
    // @ts-ignore
    const __VLS_129 = __VLS_asFunctionalComponent1(__VLS_128, new __VLS_128({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.recommendStatus),
    }));
    const __VLS_130 = __VLS_129({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.recommendStatus),
    }, ...__VLS_functionalComponentArgsRest(__VLS_129));
    let __VLS_133;
    const __VLS_134 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handleRecommendStatusStatusChange(scope.$index, scope.row);
                // @ts-ignore
                [handleRecommendStatusStatusChange,];
            } });
    var __VLS_131;
    var __VLS_132;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_124;
let __VLS_135;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_136 = __VLS_asFunctionalComponent1(__VLS_135, new __VLS_135({
    label: "排序",
    width: "160",
    align: "center",
}));
const __VLS_137 = __VLS_136({
    label: "排序",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_136));
const { default: __VLS_140 } = __VLS_138.slots;
{
    const { default: __VLS_141 } = __VLS_138.slots;
    const [scope] = __VLS_vSlot(__VLS_141);
    (scope.row.sort);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_138;
let __VLS_142;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_143 = __VLS_asFunctionalComponent1(__VLS_142, new __VLS_142({
    label: "状态",
    width: "160",
    align: "center",
}));
const __VLS_144 = __VLS_143({
    label: "状态",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_143));
const { default: __VLS_147 } = __VLS_145.slots;
{
    const { default: __VLS_148 } = __VLS_145.slots;
    const [scope] = __VLS_vSlot(__VLS_148);
    (__VLS_ctx.formatRecommendStatus(scope.row.recommendStatus));
    // @ts-ignore
    [formatRecommendStatus,];
}
// @ts-ignore
[];
var __VLS_145;
let __VLS_149;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_150 = __VLS_asFunctionalComponent1(__VLS_149, new __VLS_149({
    label: "操作",
    width: "180",
    align: "center",
}));
const __VLS_151 = __VLS_150({
    label: "操作",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_150));
const { default: __VLS_154 } = __VLS_152.slots;
{
    const { default: __VLS_155 } = __VLS_152.slots;
    const [scope] = __VLS_vSlot(__VLS_155);
    let __VLS_156;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_157 = __VLS_asFunctionalComponent1(__VLS_156, new __VLS_156({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_158 = __VLS_157({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_157));
    let __VLS_161;
    const __VLS_162 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleEditSort(scope.$index, scope.row);
                // @ts-ignore
                [handleEditSort,];
            } });
    const { default: __VLS_163 } = __VLS_159.slots;
    // @ts-ignore
    [];
    var __VLS_159;
    var __VLS_160;
    let __VLS_164;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_165 = __VLS_asFunctionalComponent1(__VLS_164, new __VLS_164({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_166 = __VLS_165({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_165));
    let __VLS_169;
    const __VLS_170 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_171 } = __VLS_167.slots;
    // @ts-ignore
    [];
    var __VLS_167;
    var __VLS_168;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_152;
// @ts-ignore
[];
var __VLS_95;
var __VLS_96;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "batch-operate-container" },
});
/** @type {__VLS_StyleScopedClasses['batch-operate-container']} */ ;
let __VLS_172;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_173 = __VLS_asFunctionalComponent1(__VLS_172, new __VLS_172({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}));
const __VLS_174 = __VLS_173({
    modelValue: (__VLS_ctx.operateType),
    placeholder: "批量操作",
}, ...__VLS_functionalComponentArgsRest(__VLS_173));
const { default: __VLS_177 } = __VLS_175.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.operates))) {
    let __VLS_178;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_179 = __VLS_asFunctionalComponent1(__VLS_178, new __VLS_178({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_180 = __VLS_179({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_179));
    // @ts-ignore
    [operateType, operates,];
}
// @ts-ignore
[];
var __VLS_175;
let __VLS_183;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_184 = __VLS_asFunctionalComponent1(__VLS_183, new __VLS_183({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}));
const __VLS_185 = __VLS_184({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_184));
let __VLS_188;
const __VLS_189 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleBatchOperate();
            // @ts-ignore
            [handleBatchOperate,];
        } });
/** @type {__VLS_StyleScopedClasses['search-button']} */ ;
const { default: __VLS_190 } = __VLS_186.slots;
// @ts-ignore
[];
var __VLS_186;
var __VLS_187;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_191;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_192 = __VLS_asFunctionalComponent1(__VLS_191, new __VLS_191({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    currentPage: (__VLS_ctx.listQuery.pageNum),
    total: (__VLS_ctx.total),
}));
const __VLS_193 = __VLS_192({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    currentPage: (__VLS_ctx.listQuery.pageNum),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_192));
let __VLS_196;
const __VLS_197 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_198 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_194;
var __VLS_195;
let __VLS_199;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_200 = __VLS_asFunctionalComponent1(__VLS_199, new __VLS_199({
    title: "选择品牌",
    modelValue: (__VLS_ctx.selectDialogVisible),
    width: "40%",
}));
const __VLS_201 = __VLS_200({
    title: "选择品牌",
    modelValue: (__VLS_ctx.selectDialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_200));
const { default: __VLS_204 } = __VLS_202.slots;
let __VLS_205;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_206 = __VLS_asFunctionalComponent1(__VLS_205, new __VLS_205({
    modelValue: (__VLS_ctx.dialogData.listQuery.keyword),
    ...{ style: {} },
    placeholder: "品牌名称搜索",
}));
const __VLS_207 = __VLS_206({
    modelValue: (__VLS_ctx.dialogData.listQuery.keyword),
    ...{ style: {} },
    placeholder: "品牌名称搜索",
}, ...__VLS_functionalComponentArgsRest(__VLS_206));
const { default: __VLS_210 } = __VLS_208.slots;
{
    const { append: __VLS_211 } = __VLS_208.slots;
    let __VLS_212;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_213 = __VLS_asFunctionalComponent1(__VLS_212, new __VLS_212({
        ...{ 'onClick': {} },
        icon: (__VLS_ctx.Search),
    }));
    const __VLS_214 = __VLS_213({
        ...{ 'onClick': {} },
        icon: (__VLS_ctx.Search),
    }, ...__VLS_functionalComponentArgsRest(__VLS_213));
    let __VLS_217;
    const __VLS_218 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleSelectSearch();
                // @ts-ignore
                [listQuery, listQuery, total, handleSizeChange, handleCurrentChange, selectDialogVisible, dialogData, Search, handleSelectSearch,];
            } });
    var __VLS_215;
    var __VLS_216;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_208;
let __VLS_219;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_220 = __VLS_asFunctionalComponent1(__VLS_219, new __VLS_219({
    ...{ 'onSelectionChange': {} },
    data: (__VLS_ctx.dialogData.list),
    border: true,
}));
const __VLS_221 = __VLS_220({
    ...{ 'onSelectionChange': {} },
    data: (__VLS_ctx.dialogData.list),
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_220));
let __VLS_224;
const __VLS_225 = ({ selectionChange: {} },
    { onSelectionChange: (__VLS_ctx.handleDialogSelectionChange) });
const { default: __VLS_226 } = __VLS_222.slots;
let __VLS_227;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_228 = __VLS_asFunctionalComponent1(__VLS_227, new __VLS_227({
    type: "selection",
    width: "60",
    align: "center",
}));
const __VLS_229 = __VLS_228({
    type: "selection",
    width: "60",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_228));
let __VLS_232;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_233 = __VLS_asFunctionalComponent1(__VLS_232, new __VLS_232({
    label: "品牌名称",
    align: "center",
}));
const __VLS_234 = __VLS_233({
    label: "品牌名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_233));
const { default: __VLS_237 } = __VLS_235.slots;
{
    const { default: __VLS_238 } = __VLS_235.slots;
    const [scope] = __VLS_vSlot(__VLS_238);
    (scope.row.name);
    // @ts-ignore
    [dialogData, handleDialogSelectionChange,];
}
// @ts-ignore
[];
var __VLS_235;
let __VLS_239;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_240 = __VLS_asFunctionalComponent1(__VLS_239, new __VLS_239({
    label: "相关",
    width: "220",
    align: "center",
}));
const __VLS_241 = __VLS_240({
    label: "相关",
    width: "220",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_240));
const { default: __VLS_244 } = __VLS_242.slots;
{
    const { default: __VLS_245 } = __VLS_242.slots;
    const [scope] = __VLS_vSlot(__VLS_245);
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "color-main" },
    });
    /** @type {__VLS_StyleScopedClasses['color-main']} */ ;
    (scope.row.productCount);
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "color-main" },
    });
    /** @type {__VLS_StyleScopedClasses['color-main']} */ ;
    (scope.row.productCommentCount);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_242;
// @ts-ignore
[];
var __VLS_222;
var __VLS_223;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_246;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_247 = __VLS_asFunctionalComponent1(__VLS_246, new __VLS_246({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "prev, pager, next",
    currentPage: (__VLS_ctx.dialogData.listQuery.pageNum),
    pageSize: (__VLS_ctx.dialogData.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.dialogData.total),
}));
const __VLS_248 = __VLS_247({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "prev, pager, next",
    currentPage: (__VLS_ctx.dialogData.listQuery.pageNum),
    pageSize: (__VLS_ctx.dialogData.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.dialogData.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_247));
let __VLS_251;
const __VLS_252 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleDialogSizeChange) });
const __VLS_253 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleDialogCurrentChange) });
var __VLS_249;
var __VLS_250;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
{
    const { footer: __VLS_254 } = __VLS_202.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
    let __VLS_255;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_256 = __VLS_asFunctionalComponent1(__VLS_255, new __VLS_255({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_257 = __VLS_256({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_256));
    let __VLS_260;
    const __VLS_261 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.selectDialogVisible = false;
                // @ts-ignore
                [selectDialogVisible, dialogData, dialogData, dialogData, handleDialogSizeChange, handleDialogCurrentChange,];
            } });
    const { default: __VLS_262 } = __VLS_258.slots;
    // @ts-ignore
    [];
    var __VLS_258;
    var __VLS_259;
    let __VLS_263;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_264 = __VLS_asFunctionalComponent1(__VLS_263, new __VLS_263({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
    }));
    const __VLS_265 = __VLS_264({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_264));
    let __VLS_268;
    const __VLS_269 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleSelectDialogConfirm();
                // @ts-ignore
                [handleSelectDialogConfirm,];
            } });
    const { default: __VLS_270 } = __VLS_266.slots;
    // @ts-ignore
    [];
    var __VLS_266;
    var __VLS_267;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_202;
let __VLS_271;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_272 = __VLS_asFunctionalComponent1(__VLS_271, new __VLS_271({
    title: "设置排序",
    modelValue: (__VLS_ctx.sortDialogVisible),
    width: "40%",
}));
const __VLS_273 = __VLS_272({
    title: "设置排序",
    modelValue: (__VLS_ctx.sortDialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_272));
const { default: __VLS_276 } = __VLS_274.slots;
let __VLS_277;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_278 = __VLS_asFunctionalComponent1(__VLS_277, new __VLS_277({
    model: (__VLS_ctx.sortDialogData),
    labelWidth: "150px",
}));
const __VLS_279 = __VLS_278({
    model: (__VLS_ctx.sortDialogData),
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_278));
const { default: __VLS_282 } = __VLS_280.slots;
let __VLS_283;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_284 = __VLS_asFunctionalComponent1(__VLS_283, new __VLS_283({
    label: "排序：",
}));
const __VLS_285 = __VLS_284({
    label: "排序：",
}, ...__VLS_functionalComponentArgsRest(__VLS_284));
const { default: __VLS_288 } = __VLS_286.slots;
let __VLS_289;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_290 = __VLS_asFunctionalComponent1(__VLS_289, new __VLS_289({
    modelValue: (__VLS_ctx.sortDialogData.sort),
    ...{ style: {} },
}));
const __VLS_291 = __VLS_290({
    modelValue: (__VLS_ctx.sortDialogData.sort),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_290));
// @ts-ignore
[sortDialogVisible, sortDialogData, sortDialogData,];
var __VLS_286;
// @ts-ignore
[];
var __VLS_280;
{
    const { footer: __VLS_294 } = __VLS_274.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
    let __VLS_295;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_296 = __VLS_asFunctionalComponent1(__VLS_295, new __VLS_295({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_297 = __VLS_296({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_296));
    let __VLS_300;
    const __VLS_301 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.sortDialogVisible = false;
                // @ts-ignore
                [sortDialogVisible,];
            } });
    const { default: __VLS_302 } = __VLS_298.slots;
    // @ts-ignore
    [];
    var __VLS_298;
    var __VLS_299;
    let __VLS_303;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_304 = __VLS_asFunctionalComponent1(__VLS_303, new __VLS_303({
        ...{ 'onClick': {} },
        type: "primary",
        size: "small",
    }));
    const __VLS_305 = __VLS_304({
        ...{ 'onClick': {} },
        type: "primary",
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_304));
    let __VLS_308;
    const __VLS_309 = ({ click: {} },
        { onClick: (__VLS_ctx.handleUpdateSort) });
    const { default: __VLS_310 } = __VLS_306.slots;
    // @ts-ignore
    [handleUpdateSort,];
    var __VLS_306;
    var __VLS_307;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_274;
// @ts-ignore
var __VLS_100 = __VLS_99;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
