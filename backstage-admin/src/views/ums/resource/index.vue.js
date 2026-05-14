/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { Search } from '@element-plus/icons-vue';
import { dayjs, ElMessage, ElMessageBox } from 'element-plus';
import { getResourceListAPI, resourceCreateAPI, resourceUpdateAPI, resourceDeleteByIdAPI } from '@/apis/resource';
import { resourceCategoryListAllAPI } from '@/apis/resourceCategory';
// 获取路由信息
const router = useRouter();
// 列表查询参数
const listQuery = ref({
    pageNum: 1,
    pageSize: 10,
});
// 资源列表数据
const list = ref([]);
// 分页总数
const total = ref(0);
// 加载状态
const listLoading = ref(false);
// 获取列表数据
const getList = async () => {
    listLoading.value = true;
    try {
        const res = await getResourceListAPI(listQuery.value);
        listLoading.value = false;
        list.value = res.data.list;
        total.value = res.data.total;
    }
    catch (error) {
        listLoading.value = false;
        console.error('获取资源列表失败:', error);
    }
};
// 分类筛选选项
const categoryOptions = ref([]);
// 默认选择分类ID
const defaultCategoryId = ref();
// 获取分类列表
const getCateList = async () => {
    try {
        const res = await resourceCategoryListAllAPI();
        const cateList = res.data;
        cateList.forEach(item => {
            categoryOptions.value.push({ label: item.name, value: item.id });
        });
        if (cateList && cateList.length > 0) {
            defaultCategoryId.value = cateList[0].id;
        }
    }
    catch (error) {
        console.error('获取资源分类失败:', error);
    }
};
// 组件挂载时获取数据
onMounted(() => {
    getList();
    getCateList();
});
// 默认资源对象
const defaultResource = {
    name: '',
    url: '',
    description: '',
    categoryId: 0
};
// 当前操作资源
const resource = ref(Object.assign({}, defaultResource));
// 编辑弹框是否可见
const dialogVisible = ref(false);
// 是否为编辑状态
const isEdit = ref(false);
// 重置搜索条件
const handleResetSearch = () => {
    listQuery.value = {
        pageNum: 1,
        pageSize: 10,
    };
};
// 搜索列表
const handleSearchList = () => {
    listQuery.value.pageNum = 1;
    getList();
};
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
// 添加资源
const handleAdd = () => {
    dialogVisible.value = true;
    isEdit.value = false;
    resource.value = Object.assign({}, defaultResource);
    if (defaultCategoryId.value) {
        resource.value.categoryId = defaultCategoryId.value;
    }
};
// 删除资源
const handleDelete = async (index, row) => {
    await ElMessageBox.confirm('是否要删除该资源?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
    });
    try {
        await resourceDeleteByIdAPI(row.id);
        ElMessage.success('删除成功!');
        getList();
    }
    catch (error) {
        console.error('删除资源失败:', error);
    }
};
// 更新资源
const handleUpdate = (index, row) => {
    dialogVisible.value = true;
    isEdit.value = true;
    resource.value = Object.assign({}, row);
};
// 处理对话框确认
const handleDialogConfirm = async () => {
    await ElMessageBox.confirm('是否要确认?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
    });
    try {
        if (isEdit.value) {
            await resourceUpdateAPI(resource.value.id, resource.value);
            ElMessage.success('修改成功！');
            dialogVisible.value = false;
            getList();
        }
        else {
            await resourceCreateAPI(resource.value);
            ElMessage.success('添加成功！');
            dialogVisible.value = false;
            getList();
        }
    }
    catch (error) {
        console.error('操作失败:', error);
    }
};
// 显示资源分类
const handleShowCategory = () => {
    router.push({ path: '/ums/resourceCategory' });
};
// 日期格式化过滤器函数
const formatDateTime = (time) => {
    if (!time) {
        return 'N/A';
    }
    return dayjs(time).format('YYYY-MM-DD HH:mm:ss');
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
    label: "资源名称：",
}));
const __VLS_41 = __VLS_40({
    label: "资源名称：",
}, ...__VLS_functionalComponentArgsRest(__VLS_40));
const { default: __VLS_44 } = __VLS_42.slots;
let __VLS_45;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_46 = __VLS_asFunctionalComponent1(__VLS_45, new __VLS_45({
    modelValue: (__VLS_ctx.listQuery.nameKeyword),
    ...{ class: "input-width" },
    placeholder: "资源名称",
    clearable: true,
}));
const __VLS_47 = __VLS_46({
    modelValue: (__VLS_ctx.listQuery.nameKeyword),
    ...{ class: "input-width" },
    placeholder: "资源名称",
    clearable: true,
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
    label: "资源路径：",
}));
const __VLS_52 = __VLS_51({
    label: "资源路径：",
}, ...__VLS_functionalComponentArgsRest(__VLS_51));
const { default: __VLS_55 } = __VLS_53.slots;
let __VLS_56;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
    modelValue: (__VLS_ctx.listQuery.urlKeyword),
    ...{ class: "input-width" },
    placeholder: "资源路径",
    clearable: true,
}));
const __VLS_58 = __VLS_57({
    modelValue: (__VLS_ctx.listQuery.urlKeyword),
    ...{ class: "input-width" },
    placeholder: "资源路径",
    clearable: true,
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
    label: "资源分类：",
}));
const __VLS_63 = __VLS_62({
    label: "资源分类：",
}, ...__VLS_functionalComponentArgsRest(__VLS_62));
const { default: __VLS_66 } = __VLS_64.slots;
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    modelValue: (__VLS_ctx.listQuery.categoryId),
    placeholder: "全部",
    clearable: true,
    ...{ class: "input-width" },
    ...{ style: {} },
}));
const __VLS_69 = __VLS_68({
    modelValue: (__VLS_ctx.listQuery.categoryId),
    placeholder: "全部",
    clearable: true,
    ...{ class: "input-width" },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
const { default: __VLS_72 } = __VLS_70.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.categoryOptions))) {
    let __VLS_73;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_74 = __VLS_asFunctionalComponent1(__VLS_73, new __VLS_73({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_75 = __VLS_74({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_74));
    // @ts-ignore
    [listQuery, categoryOptions,];
}
// @ts-ignore
[];
var __VLS_70;
// @ts-ignore
[];
var __VLS_64;
// @ts-ignore
[];
var __VLS_36;
// @ts-ignore
[];
var __VLS_3;
let __VLS_78;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_79 = __VLS_asFunctionalComponent1(__VLS_78, new __VLS_78({
    ...{ class: "operate-container" },
    shadow: "never",
}));
const __VLS_80 = __VLS_79({
    ...{ class: "operate-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_79));
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
const { default: __VLS_83 } = __VLS_81.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.i, __VLS_intrinsics.i)({
    ...{ class: "el-icon-tickets" },
});
/** @type {__VLS_StyleScopedClasses['el-icon-tickets']} */ ;
let __VLS_84;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_85 = __VLS_asFunctionalComponent1(__VLS_84, new __VLS_84({
    ...{ class: "el-icon-middle" },
}));
const __VLS_86 = __VLS_85({
    ...{ class: "el-icon-middle" },
}, ...__VLS_functionalComponentArgsRest(__VLS_85));
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_89 } = __VLS_87.slots;
let __VLS_90;
/** @ts-ignore @type { | typeof __VLS_components.tickets | typeof __VLS_components.Tickets} */
tickets;
// @ts-ignore
const __VLS_91 = __VLS_asFunctionalComponent1(__VLS_90, new __VLS_90({}));
const __VLS_92 = __VLS_91({}, ...__VLS_functionalComponentArgsRest(__VLS_91));
// @ts-ignore
[];
var __VLS_87;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
let __VLS_95;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_96 = __VLS_asFunctionalComponent1(__VLS_95, new __VLS_95({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
    ...{ style: {} },
}));
const __VLS_97 = __VLS_96({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_96));
let __VLS_100;
const __VLS_101 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleAdd();
            // @ts-ignore
            [handleAdd,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
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
    ...{ class: "btn-add" },
}));
const __VLS_105 = __VLS_104({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}, ...__VLS_functionalComponentArgsRest(__VLS_104));
let __VLS_108;
const __VLS_109 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleShowCategory();
            // @ts-ignore
            [handleShowCategory,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_110 } = __VLS_106.slots;
// @ts-ignore
[];
var __VLS_106;
var __VLS_107;
// @ts-ignore
[];
var __VLS_81;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_111;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_112 = __VLS_asFunctionalComponent1(__VLS_111, new __VLS_111({
    ref: "resourceTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_113 = __VLS_112({
    ref: "resourceTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_112));
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_116 = {};
const { default: __VLS_118 } = __VLS_114.slots;
let __VLS_119;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_120 = __VLS_asFunctionalComponent1(__VLS_119, new __VLS_119({
    label: "编号",
    width: "100",
    align: "center",
}));
const __VLS_121 = __VLS_120({
    label: "编号",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_120));
const { default: __VLS_124 } = __VLS_122.slots;
{
    const { default: __VLS_125 } = __VLS_122.slots;
    const [scope] = __VLS_vSlot(__VLS_125);
    (scope.row.id);
    // @ts-ignore
    [list, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_122;
let __VLS_126;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_127 = __VLS_asFunctionalComponent1(__VLS_126, new __VLS_126({
    label: "资源名称",
    align: "center",
}));
const __VLS_128 = __VLS_127({
    label: "资源名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_127));
const { default: __VLS_131 } = __VLS_129.slots;
{
    const { default: __VLS_132 } = __VLS_129.slots;
    const [scope] = __VLS_vSlot(__VLS_132);
    (scope.row.name);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_129;
let __VLS_133;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_134 = __VLS_asFunctionalComponent1(__VLS_133, new __VLS_133({
    label: "资源路径",
    align: "center",
}));
const __VLS_135 = __VLS_134({
    label: "资源路径",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_134));
const { default: __VLS_138 } = __VLS_136.slots;
{
    const { default: __VLS_139 } = __VLS_136.slots;
    const [scope] = __VLS_vSlot(__VLS_139);
    (scope.row.url);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_136;
let __VLS_140;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_141 = __VLS_asFunctionalComponent1(__VLS_140, new __VLS_140({
    label: "描述",
    align: "center",
}));
const __VLS_142 = __VLS_141({
    label: "描述",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_141));
const { default: __VLS_145 } = __VLS_143.slots;
{
    const { default: __VLS_146 } = __VLS_143.slots;
    const [scope] = __VLS_vSlot(__VLS_146);
    (scope.row.description);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_143;
let __VLS_147;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_148 = __VLS_asFunctionalComponent1(__VLS_147, new __VLS_147({
    label: "添加时间",
    width: "160",
    align: "center",
}));
const __VLS_149 = __VLS_148({
    label: "添加时间",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_148));
const { default: __VLS_152 } = __VLS_150.slots;
{
    const { default: __VLS_153 } = __VLS_150.slots;
    const [scope] = __VLS_vSlot(__VLS_153);
    (__VLS_ctx.formatDateTime(scope.row.createTime));
    // @ts-ignore
    [formatDateTime,];
}
// @ts-ignore
[];
var __VLS_150;
let __VLS_154;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_155 = __VLS_asFunctionalComponent1(__VLS_154, new __VLS_154({
    label: "操作",
    width: "140",
    align: "center",
}));
const __VLS_156 = __VLS_155({
    label: "操作",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_155));
const { default: __VLS_159 } = __VLS_157.slots;
{
    const { default: __VLS_160 } = __VLS_157.slots;
    const [scope] = __VLS_vSlot(__VLS_160);
    let __VLS_161;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_162 = __VLS_asFunctionalComponent1(__VLS_161, new __VLS_161({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_163 = __VLS_162({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_162));
    let __VLS_166;
    const __VLS_167 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdate(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdate,];
            } });
    const { default: __VLS_168 } = __VLS_164.slots;
    // @ts-ignore
    [];
    var __VLS_164;
    var __VLS_165;
    let __VLS_169;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_170 = __VLS_asFunctionalComponent1(__VLS_169, new __VLS_169({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_171 = __VLS_170({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_170));
    let __VLS_174;
    const __VLS_175 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_176 } = __VLS_172.slots;
    // @ts-ignore
    [];
    var __VLS_172;
    var __VLS_173;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_157;
// @ts-ignore
[];
var __VLS_114;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_177;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_178 = __VLS_asFunctionalComponent1(__VLS_177, new __VLS_177({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([10, 15, 20]),
    total: (__VLS_ctx.total),
}));
const __VLS_179 = __VLS_178({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([10, 15, 20]),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_178));
let __VLS_182;
const __VLS_183 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_184 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_180;
var __VLS_181;
let __VLS_185;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_186 = __VLS_asFunctionalComponent1(__VLS_185, new __VLS_185({
    title: (__VLS_ctx.isEdit ? '编辑资源' : '添加资源'),
    modelValue: (__VLS_ctx.dialogVisible),
    width: "40%",
}));
const __VLS_187 = __VLS_186({
    title: (__VLS_ctx.isEdit ? '编辑资源' : '添加资源'),
    modelValue: (__VLS_ctx.dialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_186));
const { default: __VLS_190 } = __VLS_188.slots;
let __VLS_191;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_192 = __VLS_asFunctionalComponent1(__VLS_191, new __VLS_191({
    model: (__VLS_ctx.resource),
    ref: "resourceForm",
    labelWidth: "150px",
}));
const __VLS_193 = __VLS_192({
    model: (__VLS_ctx.resource),
    ref: "resourceForm",
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_192));
var __VLS_196 = {};
const { default: __VLS_198 } = __VLS_194.slots;
let __VLS_199;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_200 = __VLS_asFunctionalComponent1(__VLS_199, new __VLS_199({
    label: "资源名称：",
}));
const __VLS_201 = __VLS_200({
    label: "资源名称：",
}, ...__VLS_functionalComponentArgsRest(__VLS_200));
const { default: __VLS_204 } = __VLS_202.slots;
let __VLS_205;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_206 = __VLS_asFunctionalComponent1(__VLS_205, new __VLS_205({
    modelValue: (__VLS_ctx.resource.name),
    ...{ style: {} },
}));
const __VLS_207 = __VLS_206({
    modelValue: (__VLS_ctx.resource.name),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_206));
// @ts-ignore
[listQuery, listQuery, total, handleSizeChange, handleCurrentChange, isEdit, dialogVisible, resource, resource,];
var __VLS_202;
let __VLS_210;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_211 = __VLS_asFunctionalComponent1(__VLS_210, new __VLS_210({
    label: "资源路径：",
}));
const __VLS_212 = __VLS_211({
    label: "资源路径：",
}, ...__VLS_functionalComponentArgsRest(__VLS_211));
const { default: __VLS_215 } = __VLS_213.slots;
let __VLS_216;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_217 = __VLS_asFunctionalComponent1(__VLS_216, new __VLS_216({
    modelValue: (__VLS_ctx.resource.url),
    ...{ style: {} },
}));
const __VLS_218 = __VLS_217({
    modelValue: (__VLS_ctx.resource.url),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_217));
// @ts-ignore
[resource,];
var __VLS_213;
let __VLS_221;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_222 = __VLS_asFunctionalComponent1(__VLS_221, new __VLS_221({
    label: "资源分类：",
}));
const __VLS_223 = __VLS_222({
    label: "资源分类：",
}, ...__VLS_functionalComponentArgsRest(__VLS_222));
const { default: __VLS_226 } = __VLS_224.slots;
let __VLS_227;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_228 = __VLS_asFunctionalComponent1(__VLS_227, new __VLS_227({
    modelValue: (__VLS_ctx.resource.categoryId),
    placeholder: "全部",
    clearable: true,
    ...{ style: {} },
}));
const __VLS_229 = __VLS_228({
    modelValue: (__VLS_ctx.resource.categoryId),
    placeholder: "全部",
    clearable: true,
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_228));
const { default: __VLS_232 } = __VLS_230.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.categoryOptions))) {
    let __VLS_233;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_234 = __VLS_asFunctionalComponent1(__VLS_233, new __VLS_233({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_235 = __VLS_234({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_234));
    // @ts-ignore
    [categoryOptions, resource,];
}
// @ts-ignore
[];
var __VLS_230;
// @ts-ignore
[];
var __VLS_224;
let __VLS_238;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_239 = __VLS_asFunctionalComponent1(__VLS_238, new __VLS_238({
    label: "描述：",
}));
const __VLS_240 = __VLS_239({
    label: "描述：",
}, ...__VLS_functionalComponentArgsRest(__VLS_239));
const { default: __VLS_243 } = __VLS_241.slots;
let __VLS_244;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_245 = __VLS_asFunctionalComponent1(__VLS_244, new __VLS_244({
    modelValue: (__VLS_ctx.resource.description),
    type: "textarea",
    rows: (5),
    ...{ style: {} },
}));
const __VLS_246 = __VLS_245({
    modelValue: (__VLS_ctx.resource.description),
    type: "textarea",
    rows: (5),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_245));
// @ts-ignore
[resource,];
var __VLS_241;
// @ts-ignore
[];
var __VLS_194;
{
    const { footer: __VLS_249 } = __VLS_188.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_250;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_251 = __VLS_asFunctionalComponent1(__VLS_250, new __VLS_250({
        ...{ 'onClick': {} },
    }));
    const __VLS_252 = __VLS_251({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_251));
    let __VLS_255;
    const __VLS_256 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.dialogVisible = false;
                // @ts-ignore
                [dialogVisible,];
            } });
    const { default: __VLS_257 } = __VLS_253.slots;
    // @ts-ignore
    [];
    var __VLS_253;
    var __VLS_254;
    let __VLS_258;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_259 = __VLS_asFunctionalComponent1(__VLS_258, new __VLS_258({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_260 = __VLS_259({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_259));
    let __VLS_263;
    const __VLS_264 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDialogConfirm();
                // @ts-ignore
                [handleDialogConfirm,];
            } });
    const { default: __VLS_265 } = __VLS_261.slots;
    // @ts-ignore
    [];
    var __VLS_261;
    var __VLS_262;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_188;
// @ts-ignore
var __VLS_117 = __VLS_116, __VLS_197 = __VLS_196;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
