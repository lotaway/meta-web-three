/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { dayjs, ElMessage, ElMessageBox } from 'element-plus';
import { Search, Tickets } from '@element-plus/icons-vue';
import { getRoleListAPI, roleCreateAPI, roleUpdateByIdAPI, roleUpdateStatusAPI, roleDeleteByIdsAPI } from '@/apis/role';
import { t } from '@/locales';
const router = useRouter();
const listQuery = ref({
    pageNum: 1,
    pageSize: 10,
    keyword: ''
});
const list = ref([]);
const listLoading = ref(false);
const total = ref(0);
const getList = async () => {
    listLoading.value = true;
    try {
        const response = await getRoleListAPI(listQuery.value);
        listLoading.value = false;
        list.value = response.data.list;
        total.value = response.data.total;
    }
    catch (error) {
        listLoading.value = false;
        console.error('获取角色列表失败:', error);
    }
};
onMounted(() => {
    getList();
});
const role = ref({
    name: '',
    adminCount: 0,
    status: 1
});
const dialogVisible = ref(false);
const isEdit = ref(false);
const handleResetSearch = () => {
    listQuery.value.pageNum = 1;
    listQuery.value.keyword = '';
};
const handleSearchList = () => {
    listQuery.value.pageNum = 1;
    getList();
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
const handleAdd = () => {
    dialogVisible.value = true;
    isEdit.value = false;
    role.value = {
        name: '',
        adminCount: 0,
        status: 1
    };
};
const handleStatusChange = async (index, row) => {
    try {
        await ElMessageBox.confirm('是否要修改该状态?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await roleUpdateStatusAPI(row.id, { status: row.status });
        ElMessage.success('修改成功!');
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('更新状态失败:', error);
            getList();
        }
    }
};
const handleDelete = async (index, row) => {
    try {
        await ElMessageBox.confirm('是否要删除该角色?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        const ids = [];
        ids.push(row.id);
        await roleDeleteByIdsAPI({ ids: ids.toString() });
        ElMessage.success('删除成功!');
        getList();
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('删除失败:', error);
        }
    }
};
const handleUpdate = (index, row) => {
    dialogVisible.value = true;
    isEdit.value = true;
    role.value = { ...row };
};
const handleDialogConfirm = async () => {
    try {
        await ElMessageBox.confirm('是否要确认?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        if (isEdit.value) {
            await roleUpdateByIdAPI(role.value.id, role.value);
            ElMessage.success('修改成功！');
        }
        else {
            await roleCreateAPI(role.value);
            ElMessage.success('添加成功！');
        }
        dialogVisible.value = false;
        getList();
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('操作失败:', error);
        }
    }
};
const handleSelectMenu = (index, row) => {
    router.push({ path: '/ums/allocMenu', query: { roleId: row.id } });
};
const handleSelectResource = (index, row) => {
    router.push({ path: '/ums/allocResource', query: { roleId: row.id } });
};
const formatDateTime = (time) => {
    if (!time) {
        return 'N/A';
    }
    return dayjs(time).format('YYYY-MM-DD HH:mm:ss');
};
const getRoleName = (name) => {
    return t(`role.${name}`) || name;
};
const getRoleDescription = (description) => {
    return t(`role.${description}`) || description;
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
    size: "default",
}));
const __VLS_19 = __VLS_18({
    ...{ 'onClick': {} },
    ...{ style: {} },
    type: "primary",
    size: "default",
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
    size: "default",
}));
const __VLS_27 = __VLS_26({
    ...{ 'onClick': {} },
    ...{ style: {} },
    size: "default",
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
    size: "default",
    labelWidth: "140px",
}));
const __VLS_35 = __VLS_34({
    inline: (true),
    model: (__VLS_ctx.listQuery),
    size: "default",
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
    modelValue: (__VLS_ctx.listQuery.keyword),
    ...{ class: "input-width" },
    placeholder: "角色名称",
    clearable: true,
}));
const __VLS_47 = __VLS_46({
    modelValue: (__VLS_ctx.listQuery.keyword),
    ...{ class: "input-width" },
    placeholder: "角色名称",
    clearable: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_46));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery, listQuery,];
var __VLS_42;
// @ts-ignore
[];
var __VLS_36;
// @ts-ignore
[];
var __VLS_3;
let __VLS_50;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_51 = __VLS_asFunctionalComponent1(__VLS_50, new __VLS_50({
    ...{ class: "operate-container" },
    shadow: "never",
}));
const __VLS_52 = __VLS_51({
    ...{ class: "operate-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_51));
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
const { default: __VLS_55 } = __VLS_53.slots;
let __VLS_56;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
    ...{ class: "el-icon-middle" },
}));
const __VLS_58 = __VLS_57({
    ...{ class: "el-icon-middle" },
}, ...__VLS_functionalComponentArgsRest(__VLS_57));
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_61 } = __VLS_59.slots;
let __VLS_62;
/** @ts-ignore @type { | typeof __VLS_components.Tickets} */
Tickets;
// @ts-ignore
const __VLS_63 = __VLS_asFunctionalComponent1(__VLS_62, new __VLS_62({}));
const __VLS_64 = __VLS_63({}, ...__VLS_functionalComponentArgsRest(__VLS_63));
// @ts-ignore
[];
var __VLS_59;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    ...{ 'onClick': {} },
    size: "default",
    ...{ class: "btn-add" },
    ...{ style: {} },
}));
const __VLS_69 = __VLS_68({
    ...{ 'onClick': {} },
    size: "default",
    ...{ class: "btn-add" },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
let __VLS_72;
const __VLS_73 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleAdd();
            // @ts-ignore
            [handleAdd,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_74 } = __VLS_70.slots;
// @ts-ignore
[];
var __VLS_70;
var __VLS_71;
// @ts-ignore
[];
var __VLS_53;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_75;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_76 = __VLS_asFunctionalComponent1(__VLS_75, new __VLS_75({
    ref: "roleTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_77 = __VLS_76({
    ref: "roleTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_76));
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_80 = {};
const { default: __VLS_82 } = __VLS_78.slots;
let __VLS_83;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_84 = __VLS_asFunctionalComponent1(__VLS_83, new __VLS_83({
    label: "编号",
    width: "100",
    align: "center",
}));
const __VLS_85 = __VLS_84({
    label: "编号",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_84));
const { default: __VLS_88 } = __VLS_86.slots;
{
    const { default: __VLS_89 } = __VLS_86.slots;
    const [scope] = __VLS_vSlot(__VLS_89);
    (scope.row.id);
    // @ts-ignore
    [list, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_86;
let __VLS_90;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_91 = __VLS_asFunctionalComponent1(__VLS_90, new __VLS_90({
    label: "角色名称",
    align: "center",
}));
const __VLS_92 = __VLS_91({
    label: "角色名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_91));
const { default: __VLS_95 } = __VLS_93.slots;
{
    const { default: __VLS_96 } = __VLS_93.slots;
    const [scope] = __VLS_vSlot(__VLS_96);
    (__VLS_ctx.getRoleName(scope.row.name));
    // @ts-ignore
    [getRoleName,];
}
// @ts-ignore
[];
var __VLS_93;
let __VLS_97;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_98 = __VLS_asFunctionalComponent1(__VLS_97, new __VLS_97({
    label: "描述",
    align: "center",
}));
const __VLS_99 = __VLS_98({
    label: "描述",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_98));
const { default: __VLS_102 } = __VLS_100.slots;
{
    const { default: __VLS_103 } = __VLS_100.slots;
    const [scope] = __VLS_vSlot(__VLS_103);
    (__VLS_ctx.getRoleDescription(scope.row.description));
    // @ts-ignore
    [getRoleDescription,];
}
// @ts-ignore
[];
var __VLS_100;
let __VLS_104;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_105 = __VLS_asFunctionalComponent1(__VLS_104, new __VLS_104({
    label: "用户数",
    width: "100",
    align: "center",
}));
const __VLS_106 = __VLS_105({
    label: "用户数",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_105));
const { default: __VLS_109 } = __VLS_107.slots;
{
    const { default: __VLS_110 } = __VLS_107.slots;
    const [scope] = __VLS_vSlot(__VLS_110);
    (scope.row.adminCount);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_107;
let __VLS_111;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_112 = __VLS_asFunctionalComponent1(__VLS_111, new __VLS_111({
    label: "添加时间",
    width: "160",
    align: "center",
}));
const __VLS_113 = __VLS_112({
    label: "添加时间",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_112));
const { default: __VLS_116 } = __VLS_114.slots;
{
    const { default: __VLS_117 } = __VLS_114.slots;
    const [scope] = __VLS_vSlot(__VLS_117);
    (__VLS_ctx.formatDateTime(scope.row.createTime));
    // @ts-ignore
    [formatDateTime,];
}
// @ts-ignore
[];
var __VLS_114;
let __VLS_118;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_119 = __VLS_asFunctionalComponent1(__VLS_118, new __VLS_118({
    label: "是否启用",
    width: "140",
    align: "center",
}));
const __VLS_120 = __VLS_119({
    label: "是否启用",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_119));
const { default: __VLS_123 } = __VLS_121.slots;
{
    const { default: __VLS_124 } = __VLS_121.slots;
    const [scope] = __VLS_vSlot(__VLS_124);
    let __VLS_125;
    /** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
    elSwitch;
    // @ts-ignore
    const __VLS_126 = __VLS_asFunctionalComponent1(__VLS_125, new __VLS_125({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.status),
    }));
    const __VLS_127 = __VLS_126({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.status),
    }, ...__VLS_functionalComponentArgsRest(__VLS_126));
    let __VLS_130;
    const __VLS_131 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handleStatusChange(scope.$index, scope.row);
                // @ts-ignore
                [handleStatusChange,];
            } });
    var __VLS_128;
    var __VLS_129;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_121;
let __VLS_132;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_133 = __VLS_asFunctionalComponent1(__VLS_132, new __VLS_132({
    label: "操作",
    width: "160",
    align: "center",
}));
const __VLS_134 = __VLS_133({
    label: "操作",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_133));
const { default: __VLS_137 } = __VLS_135.slots;
{
    const { default: __VLS_138 } = __VLS_135.slots;
    const [scope] = __VLS_vSlot(__VLS_138);
    let __VLS_139;
    /** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
    elRow;
    // @ts-ignore
    const __VLS_140 = __VLS_asFunctionalComponent1(__VLS_139, new __VLS_139({}));
    const __VLS_141 = __VLS_140({}, ...__VLS_functionalComponentArgsRest(__VLS_140));
    const { default: __VLS_144 } = __VLS_142.slots;
    let __VLS_145;
    /** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
    elCol;
    // @ts-ignore
    const __VLS_146 = __VLS_asFunctionalComponent1(__VLS_145, new __VLS_145({
        span: (12),
    }));
    const __VLS_147 = __VLS_146({
        span: (12),
    }, ...__VLS_functionalComponentArgsRest(__VLS_146));
    const { default: __VLS_150 } = __VLS_148.slots;
    let __VLS_151;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_152 = __VLS_asFunctionalComponent1(__VLS_151, new __VLS_151({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_153 = __VLS_152({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_152));
    let __VLS_156;
    const __VLS_157 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleSelectMenu(scope.$index, scope.row);
                // @ts-ignore
                [handleSelectMenu,];
            } });
    const { default: __VLS_158 } = __VLS_154.slots;
    // @ts-ignore
    [];
    var __VLS_154;
    var __VLS_155;
    // @ts-ignore
    [];
    var __VLS_148;
    let __VLS_159;
    /** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
    elCol;
    // @ts-ignore
    const __VLS_160 = __VLS_asFunctionalComponent1(__VLS_159, new __VLS_159({
        span: (12),
    }));
    const __VLS_161 = __VLS_160({
        span: (12),
    }, ...__VLS_functionalComponentArgsRest(__VLS_160));
    const { default: __VLS_164 } = __VLS_162.slots;
    let __VLS_165;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_166 = __VLS_asFunctionalComponent1(__VLS_165, new __VLS_165({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_167 = __VLS_166({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_166));
    let __VLS_170;
    const __VLS_171 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleSelectResource(scope.$index, scope.row);
                // @ts-ignore
                [handleSelectResource,];
            } });
    const { default: __VLS_172 } = __VLS_168.slots;
    // @ts-ignore
    [];
    var __VLS_168;
    var __VLS_169;
    // @ts-ignore
    [];
    var __VLS_162;
    // @ts-ignore
    [];
    var __VLS_142;
    let __VLS_173;
    /** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
    elRow;
    // @ts-ignore
    const __VLS_174 = __VLS_asFunctionalComponent1(__VLS_173, new __VLS_173({}));
    const __VLS_175 = __VLS_174({}, ...__VLS_functionalComponentArgsRest(__VLS_174));
    const { default: __VLS_178 } = __VLS_176.slots;
    let __VLS_179;
    /** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
    elCol;
    // @ts-ignore
    const __VLS_180 = __VLS_asFunctionalComponent1(__VLS_179, new __VLS_179({
        span: (12),
    }));
    const __VLS_181 = __VLS_180({
        span: (12),
    }, ...__VLS_functionalComponentArgsRest(__VLS_180));
    const { default: __VLS_184 } = __VLS_182.slots;
    let __VLS_185;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_186 = __VLS_asFunctionalComponent1(__VLS_185, new __VLS_185({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_187 = __VLS_186({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_186));
    let __VLS_190;
    const __VLS_191 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdate(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdate,];
            } });
    const { default: __VLS_192 } = __VLS_188.slots;
    // @ts-ignore
    [];
    var __VLS_188;
    var __VLS_189;
    // @ts-ignore
    [];
    var __VLS_182;
    let __VLS_193;
    /** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
    elCol;
    // @ts-ignore
    const __VLS_194 = __VLS_asFunctionalComponent1(__VLS_193, new __VLS_193({
        span: (12),
    }));
    const __VLS_195 = __VLS_194({
        span: (12),
    }, ...__VLS_functionalComponentArgsRest(__VLS_194));
    const { default: __VLS_198 } = __VLS_196.slots;
    let __VLS_199;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_200 = __VLS_asFunctionalComponent1(__VLS_199, new __VLS_199({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_201 = __VLS_200({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_200));
    let __VLS_204;
    const __VLS_205 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_206 } = __VLS_202.slots;
    // @ts-ignore
    [];
    var __VLS_202;
    var __VLS_203;
    // @ts-ignore
    [];
    var __VLS_196;
    // @ts-ignore
    [];
    var __VLS_176;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_135;
// @ts-ignore
[];
var __VLS_78;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_207;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_208 = __VLS_asFunctionalComponent1(__VLS_207, new __VLS_207({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}));
const __VLS_209 = __VLS_208({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_208));
let __VLS_212;
const __VLS_213 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_214 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_210;
var __VLS_211;
let __VLS_215;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_216 = __VLS_asFunctionalComponent1(__VLS_215, new __VLS_215({
    modelValue: (__VLS_ctx.dialogVisible),
    title: (__VLS_ctx.isEdit ? '编辑角色' : '添加角色'),
    width: "40%",
}));
const __VLS_217 = __VLS_216({
    modelValue: (__VLS_ctx.dialogVisible),
    title: (__VLS_ctx.isEdit ? '编辑角色' : '添加角色'),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_216));
const { default: __VLS_220 } = __VLS_218.slots;
let __VLS_221;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_222 = __VLS_asFunctionalComponent1(__VLS_221, new __VLS_221({
    model: (__VLS_ctx.role),
    labelWidth: "150px",
    size: "default",
}));
const __VLS_223 = __VLS_222({
    model: (__VLS_ctx.role),
    labelWidth: "150px",
    size: "default",
}, ...__VLS_functionalComponentArgsRest(__VLS_222));
const { default: __VLS_226 } = __VLS_224.slots;
let __VLS_227;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_228 = __VLS_asFunctionalComponent1(__VLS_227, new __VLS_227({
    label: "角色名称：",
}));
const __VLS_229 = __VLS_228({
    label: "角色名称：",
}, ...__VLS_functionalComponentArgsRest(__VLS_228));
const { default: __VLS_232 } = __VLS_230.slots;
let __VLS_233;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_234 = __VLS_asFunctionalComponent1(__VLS_233, new __VLS_233({
    modelValue: (__VLS_ctx.role.name),
    ...{ style: {} },
}));
const __VLS_235 = __VLS_234({
    modelValue: (__VLS_ctx.role.name),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_234));
// @ts-ignore
[listQuery, listQuery, total, handleSizeChange, handleCurrentChange, dialogVisible, isEdit, role, role,];
var __VLS_230;
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
    modelValue: (__VLS_ctx.role.description),
    type: "textarea",
    rows: (5),
    ...{ style: {} },
}));
const __VLS_246 = __VLS_245({
    modelValue: (__VLS_ctx.role.description),
    type: "textarea",
    rows: (5),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_245));
// @ts-ignore
[role,];
var __VLS_241;
let __VLS_249;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_250 = __VLS_asFunctionalComponent1(__VLS_249, new __VLS_249({
    label: "是否启用：",
}));
const __VLS_251 = __VLS_250({
    label: "是否启用：",
}, ...__VLS_functionalComponentArgsRest(__VLS_250));
const { default: __VLS_254 } = __VLS_252.slots;
let __VLS_255;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_256 = __VLS_asFunctionalComponent1(__VLS_255, new __VLS_255({
    modelValue: (__VLS_ctx.role.status),
}));
const __VLS_257 = __VLS_256({
    modelValue: (__VLS_ctx.role.status),
}, ...__VLS_functionalComponentArgsRest(__VLS_256));
const { default: __VLS_260 } = __VLS_258.slots;
let __VLS_261;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_262 = __VLS_asFunctionalComponent1(__VLS_261, new __VLS_261({
    label: (1),
}));
const __VLS_263 = __VLS_262({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_262));
const { default: __VLS_266 } = __VLS_264.slots;
// @ts-ignore
[role,];
var __VLS_264;
let __VLS_267;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_268 = __VLS_asFunctionalComponent1(__VLS_267, new __VLS_267({
    label: (0),
}));
const __VLS_269 = __VLS_268({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_268));
const { default: __VLS_272 } = __VLS_270.slots;
// @ts-ignore
[];
var __VLS_270;
// @ts-ignore
[];
var __VLS_258;
// @ts-ignore
[];
var __VLS_252;
// @ts-ignore
[];
var __VLS_224;
{
    const { footer: __VLS_273 } = __VLS_218.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_274;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_275 = __VLS_asFunctionalComponent1(__VLS_274, new __VLS_274({
        ...{ 'onClick': {} },
        size: "default",
    }));
    const __VLS_276 = __VLS_275({
        ...{ 'onClick': {} },
        size: "default",
    }, ...__VLS_functionalComponentArgsRest(__VLS_275));
    let __VLS_279;
    const __VLS_280 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.dialogVisible = false;
                // @ts-ignore
                [dialogVisible,];
            } });
    const { default: __VLS_281 } = __VLS_277.slots;
    // @ts-ignore
    [];
    var __VLS_277;
    var __VLS_278;
    let __VLS_282;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_283 = __VLS_asFunctionalComponent1(__VLS_282, new __VLS_282({
        ...{ 'onClick': {} },
        type: "primary",
        size: "default",
    }));
    const __VLS_284 = __VLS_283({
        ...{ 'onClick': {} },
        type: "primary",
        size: "default",
    }, ...__VLS_functionalComponentArgsRest(__VLS_283));
    let __VLS_287;
    const __VLS_288 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDialogConfirm();
                // @ts-ignore
                [handleDialogConfirm,];
            } });
    const { default: __VLS_289 } = __VLS_285.slots;
    // @ts-ignore
    [];
    var __VLS_285;
    var __VLS_286;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_218;
// @ts-ignore
var __VLS_81 = __VLS_80;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
