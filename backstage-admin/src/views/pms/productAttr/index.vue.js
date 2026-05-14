/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getProductAttributeCategoryListAPI, productAttributeCategoryCreateAPI, productAttributeCategoryDeleteById, productAttributeCategoryUpdateAPI } from '@/apis/productAttrCate';
import { Tickets } from '@element-plus/icons-vue';
// 获取路由对象
const router = useRouter();
// 列表查询参数
const listQuery = ref({
    pageNum: 1,
    pageSize: 10
});
// 列表数据
const list = ref([]);
// 总条数
const total = ref(0);
// 加载状态
const listLoading = ref(true);
// 获取列表数据
const getList = async () => {
    listLoading.value = true;
    try {
        const response = await getProductAttributeCategoryListAPI(listQuery.value);
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
// 当前操作的属性分类
const productAttrCate = ref({
    name: ''
});
// 编辑框是否可见
const dialogVisible = ref(false);
// 编辑框标题
const dialogTitle = ref('');
// 编辑框中的表单引用
const productAttrCatForm = ref();
// 编辑框中的校验规则
const rules = ref({
    name: [
        { required: true, message: '请输入类型名称', trigger: 'blur' }
    ]
});
// 添加产品属性分类
const addProductAttrCate = () => {
    dialogVisible.value = true;
    dialogTitle.value = "添加类型";
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
// 处理删除操作
const handleDelete = async (index, row) => {
    await ElMessageBox.confirm('是否要删除该品牌', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    await productAttributeCategoryDeleteById(row.id);
    ElMessage.success('删除成功');
    getList();
};
// 处理更新操作
const handleUpdate = (index, row) => {
    dialogVisible.value = true;
    dialogTitle.value = "编辑类型";
    productAttrCate.value.name = row.name;
    productAttrCate.value.id = row.id;
};
// 获取属性列表
const getAttrList = (index, row) => {
    router.push({ path: '/pms/productAttrList', query: { cid: row.id, cname: row.name, type: 0 } });
};
// 获取参数列表
const getParamList = (index, row) => {
    router.push({ path: '/pms/productAttrList', query: { cid: row.id, cname: row.name, type: 1 } });
};
// 确认对话框操作
const handleConfirm = async () => {
    if (!productAttrCatForm.value)
        return;
    const valid = await productAttrCatForm.value.validate().catch(() => false);
    if (valid) {
        if (dialogTitle.value === "添加类型") {
            await productAttributeCategoryCreateAPI(productAttrCate.value.name);
            ElMessage.success('添加成功');
            dialogVisible.value = false;
        }
        else {
            await productAttributeCategoryUpdateAPI(productAttrCate.value.id, productAttrCate.value.name);
            ElMessage.success('修改成功');
            dialogVisible.value = false;
        }
        getList();
    }
    else {
        ElMessage.error('请先填写类型名称！');
        return false;
    }
};
// 关闭对话框处理
const handleClose = () => {
    dialogVisible.value = false;
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
            __VLS_ctx.addProductAttrCate();
            // @ts-ignore
            [addProductAttrCate,];
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
    ref: "productAttrCateTable",
    ...{ style: {} },
    data: (__VLS_ctx.list),
    border: true,
}));
const __VLS_27 = __VLS_26({
    ref: "productAttrCateTable",
    ...{ style: {} },
    data: (__VLS_ctx.list),
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_26));
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_30 = {};
const { default: __VLS_32 } = __VLS_28.slots;
let __VLS_33;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_34 = __VLS_asFunctionalComponent1(__VLS_33, new __VLS_33({
    label: "编号",
    width: "100",
    align: "center",
}));
const __VLS_35 = __VLS_34({
    label: "编号",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_34));
const { default: __VLS_38 } = __VLS_36.slots;
{
    const { default: __VLS_39 } = __VLS_36.slots;
    const [scope] = __VLS_vSlot(__VLS_39);
    (scope.row.id);
    // @ts-ignore
    [list, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_36;
let __VLS_40;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_41 = __VLS_asFunctionalComponent1(__VLS_40, new __VLS_40({
    label: "类型名称",
    align: "center",
}));
const __VLS_42 = __VLS_41({
    label: "类型名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_41));
const { default: __VLS_45 } = __VLS_43.slots;
{
    const { default: __VLS_46 } = __VLS_43.slots;
    const [scope] = __VLS_vSlot(__VLS_46);
    (scope.row.name);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_43;
let __VLS_47;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({
    label: "属性数量",
    width: "200",
    align: "center",
}));
const __VLS_49 = __VLS_48({
    label: "属性数量",
    width: "200",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_48));
const { default: __VLS_52 } = __VLS_50.slots;
{
    const { default: __VLS_53 } = __VLS_50.slots;
    const [scope] = __VLS_vSlot(__VLS_53);
    (scope.row.attributeCount == null ? 0 : scope.row.attributeCount);
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
    label: "参数数量",
    width: "200",
    align: "center",
}));
const __VLS_56 = __VLS_55({
    label: "参数数量",
    width: "200",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
const { default: __VLS_59 } = __VLS_57.slots;
{
    const { default: __VLS_60 } = __VLS_57.slots;
    const [scope] = __VLS_vSlot(__VLS_60);
    (scope.row.paramCount == null ? 0 : scope.row.paramCount);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_57;
let __VLS_61;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_62 = __VLS_asFunctionalComponent1(__VLS_61, new __VLS_61({
    label: "设置",
    width: "200",
    align: "center",
}));
const __VLS_63 = __VLS_62({
    label: "设置",
    width: "200",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_62));
const { default: __VLS_66 } = __VLS_64.slots;
{
    const { default: __VLS_67 } = __VLS_64.slots;
    const [scope] = __VLS_vSlot(__VLS_67);
    let __VLS_68;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_69 = __VLS_asFunctionalComponent1(__VLS_68, new __VLS_68({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_70 = __VLS_69({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_69));
    let __VLS_73;
    const __VLS_74 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.getAttrList(scope.$index, scope.row);
                // @ts-ignore
                [getAttrList,];
            } });
    const { default: __VLS_75 } = __VLS_71.slots;
    // @ts-ignore
    [];
    var __VLS_71;
    var __VLS_72;
    let __VLS_76;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_77 = __VLS_asFunctionalComponent1(__VLS_76, new __VLS_76({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_78 = __VLS_77({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_77));
    let __VLS_81;
    const __VLS_82 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.getParamList(scope.$index, scope.row);
                // @ts-ignore
                [getParamList,];
            } });
    const { default: __VLS_83 } = __VLS_79.slots;
    // @ts-ignore
    [];
    var __VLS_79;
    var __VLS_80;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_64;
let __VLS_84;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_85 = __VLS_asFunctionalComponent1(__VLS_84, new __VLS_84({
    label: "操作",
    width: "200",
    align: "center",
}));
const __VLS_86 = __VLS_85({
    label: "操作",
    width: "200",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_85));
const { default: __VLS_89 } = __VLS_87.slots;
{
    const { default: __VLS_90 } = __VLS_87.slots;
    const [scope] = __VLS_vSlot(__VLS_90);
    let __VLS_91;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_92 = __VLS_asFunctionalComponent1(__VLS_91, new __VLS_91({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_93 = __VLS_92({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_92));
    let __VLS_96;
    const __VLS_97 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdate(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdate,];
            } });
    const { default: __VLS_98 } = __VLS_94.slots;
    // @ts-ignore
    [];
    var __VLS_94;
    var __VLS_95;
    let __VLS_99;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_100 = __VLS_asFunctionalComponent1(__VLS_99, new __VLS_99({
        ...{ 'onClick': {} },
        size: "small",
        type: "danger",
    }));
    const __VLS_101 = __VLS_100({
        ...{ 'onClick': {} },
        size: "small",
        type: "danger",
    }, ...__VLS_functionalComponentArgsRest(__VLS_100));
    let __VLS_104;
    const __VLS_105 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_106 } = __VLS_102.slots;
    // @ts-ignore
    [];
    var __VLS_102;
    var __VLS_103;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_87;
// @ts-ignore
[];
var __VLS_28;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_107;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_108 = __VLS_asFunctionalComponent1(__VLS_107, new __VLS_107({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    currentPage: (__VLS_ctx.listQuery.pageNum),
    total: (__VLS_ctx.total),
}));
const __VLS_109 = __VLS_108({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    currentPage: (__VLS_ctx.listQuery.pageNum),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_108));
let __VLS_112;
const __VLS_113 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_114 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_110;
var __VLS_111;
let __VLS_115;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_116 = __VLS_asFunctionalComponent1(__VLS_115, new __VLS_115({
    title: (__VLS_ctx.dialogTitle),
    modelValue: (__VLS_ctx.dialogVisible),
    beforeClose: (__VLS_ctx.handleClose),
    width: "30%",
}));
const __VLS_117 = __VLS_116({
    title: (__VLS_ctx.dialogTitle),
    modelValue: (__VLS_ctx.dialogVisible),
    beforeClose: (__VLS_ctx.handleClose),
    width: "30%",
}, ...__VLS_functionalComponentArgsRest(__VLS_116));
const { default: __VLS_120 } = __VLS_118.slots;
let __VLS_121;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_122 = __VLS_asFunctionalComponent1(__VLS_121, new __VLS_121({
    ref: "productAttrCatForm",
    model: (__VLS_ctx.productAttrCate),
    rules: (__VLS_ctx.rules),
    labelWidth: "120px",
}));
const __VLS_123 = __VLS_122({
    ref: "productAttrCatForm",
    model: (__VLS_ctx.productAttrCate),
    rules: (__VLS_ctx.rules),
    labelWidth: "120px",
}, ...__VLS_functionalComponentArgsRest(__VLS_122));
var __VLS_126 = {};
const { default: __VLS_128 } = __VLS_124.slots;
let __VLS_129;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_130 = __VLS_asFunctionalComponent1(__VLS_129, new __VLS_129({
    label: "类型名称",
    prop: "name",
}));
const __VLS_131 = __VLS_130({
    label: "类型名称",
    prop: "name",
}, ...__VLS_functionalComponentArgsRest(__VLS_130));
const { default: __VLS_134 } = __VLS_132.slots;
let __VLS_135;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_136 = __VLS_asFunctionalComponent1(__VLS_135, new __VLS_135({
    modelValue: (__VLS_ctx.productAttrCate.name),
    autocomplete: "off",
}));
const __VLS_137 = __VLS_136({
    modelValue: (__VLS_ctx.productAttrCate.name),
    autocomplete: "off",
}, ...__VLS_functionalComponentArgsRest(__VLS_136));
// @ts-ignore
[listQuery, listQuery, total, handleSizeChange, handleCurrentChange, dialogTitle, dialogVisible, handleClose, productAttrCate, productAttrCate, rules,];
var __VLS_132;
// @ts-ignore
[];
var __VLS_124;
{
    const { footer: __VLS_140 } = __VLS_118.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_141;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_142 = __VLS_asFunctionalComponent1(__VLS_141, new __VLS_141({
        ...{ 'onClick': {} },
    }));
    const __VLS_143 = __VLS_142({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_142));
    let __VLS_146;
    const __VLS_147 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.dialogVisible = false;
                // @ts-ignore
                [dialogVisible,];
            } });
    const { default: __VLS_148 } = __VLS_144.slots;
    // @ts-ignore
    [];
    var __VLS_144;
    var __VLS_145;
    let __VLS_149;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_150 = __VLS_asFunctionalComponent1(__VLS_149, new __VLS_149({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_151 = __VLS_150({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_150));
    let __VLS_154;
    const __VLS_155 = ({ click: {} },
        { onClick: (__VLS_ctx.handleConfirm) });
    const { default: __VLS_156 } = __VLS_152.slots;
    // @ts-ignore
    [handleConfirm,];
    var __VLS_152;
    var __VLS_153;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_118;
// @ts-ignore
var __VLS_31 = __VLS_30, __VLS_127 = __VLS_126;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
