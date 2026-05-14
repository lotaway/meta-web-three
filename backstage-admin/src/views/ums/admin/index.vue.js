/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { dayjs, ElMessage, ElMessageBox } from 'element-plus';
import { Search, Tickets } from '@element-plus/icons-vue';
import { getAdminListAPI, adminRegisterAPI, adminUpdateByIdAPI, adminUpdateStatusByIdAPI, adminDeleteByIdAPI, getRoleByAdminIdAPI, adminRoleUpdateAPI } from '@/apis/admin.ts';
import { getRoleListAllAPI } from '@/apis/role.ts';
// 列表查询参数
const listQuery = ref({
    pageNum: 1,
    pageSize: 10,
    keyword: ''
});
// 管理员列表数据
const list = ref([]);
// 所有角色列表
const allRoleList = ref([]);
// 表格加载状态
const listLoading = ref(true);
// 分页总数
const total = ref(0);
// 获取管理员列表
const getList = async () => {
    listLoading.value = true;
    try {
        const res = await getAdminListAPI(listQuery.value);
        listLoading.value = false;
        list.value = res.data.list;
        total.value = res.data.total;
    }
    catch (error) {
        listLoading.value = false;
        console.error('获取管理员列表失败:', error);
    }
};
// 获取所有角色列表
const getAllRoleList = async () => {
    try {
        const response = await getRoleListAllAPI();
        allRoleList.value = response.data;
    }
    catch (error) {
        console.error('获取角色列表失败:', error);
    }
};
// 组件挂载后初始化数据
onMounted(() => {
    getList();
    getAllRoleList();
});
// 当前操作的管理员
const admin = ref({
    username: '',
    password: '',
    status: 1
});
// 管理员编辑对话框是否可见
const dialogVisible = ref(false);
// 是否编辑状态
const isEdit = ref(false);
// 分配角色对话框是否可见
const allocDialogVisible = ref(false);
// 当前正在分配角色的管理员ID
const allocAdminId = ref();
// 当前管理员已分配的角色ID
const allocRoleIds = ref([]);
// 根据管理员ID获取角色列表
const getRoleListByAdmin = async (adminId) => {
    try {
        const res = await getRoleByAdminIdAPI(adminId);
        const allocRoleList = res.data;
        allocRoleIds.value = [];
        allocRoleList.forEach((item) => allocRoleIds.value.push(item.id));
    }
    catch (error) {
        console.error('获取管理员角色列表失败:', error);
    }
};
// 重置搜索条件
const handleResetSearch = () => {
    listQuery.value.pageNum = 1;
    listQuery.value.keyword = '';
};
// 处理搜索
const handleSearchList = () => {
    listQuery.value.pageNum = 1;
    getList();
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
// 处理添加管理员
const handleAdd = () => {
    dialogVisible.value = true;
    isEdit.value = false;
    admin.value = {
        username: '',
        password: '',
        status: 1
    };
};
// 处理状态变化
const handleStatusChange = async (index, row) => {
    try {
        await ElMessageBox.confirm('是否要修改该状态?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await adminUpdateStatusByIdAPI(row.id, { status: row.status });
        ElMessage.success('修改成功!');
    }
    catch (error) {
        console.error('更新状态失败:', error);
        // 如果取消或失败，重新获取列表以恢复状态
        getList();
    }
};
// 处理删除
const handleDelete = async (index, row) => {
    try {
        await ElMessageBox.confirm('是否要删除该用户?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await adminDeleteByIdAPI(row.id);
        ElMessage.success('删除成功!');
        getList();
    }
    catch (error) {
        console.error('删除失败:', error);
    }
};
// 处理更新
const handleUpdate = (index, row) => {
    dialogVisible.value = true;
    isEdit.value = true;
    // 深拷贝row对象，避免直接修改原始数据
    admin.value = Object.assign({}, row);
};
// 处理对话框确认
const handleDialogConfirm = async () => {
    try {
        await ElMessageBox.confirm('是否要确认?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        if (isEdit.value) {
            await adminUpdateByIdAPI(admin.value.id, admin.value);
            ElMessage.success('修改成功！');
        }
        else {
            await adminRegisterAPI(admin.value);
            ElMessage.success('添加成功！');
        }
        dialogVisible.value = false;
        getList();
    }
    catch (error) {
        console.error('操作失败:', error);
    }
};
// 处理分配角色对话框确认
const handleAllocDialogConfirm = async () => {
    try {
        await ElMessageBox.confirm('是否要确认?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await adminRoleUpdateAPI({ adminId: allocAdminId.value, roleIds: allocRoleIds.value.join(',') });
        ElMessage.success('分配成功！');
        allocDialogVisible.value = false;
    }
    catch (error) {
        console.error('分配角色失败:', error);
    }
};
// 处理选择角色
const handleSelectRole = (index, row) => {
    allocAdminId.value = row.id;
    allocDialogVisible.value = true;
    getRoleListByAdmin(allocAdminId.value);
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
    placeholder: "帐号/姓名",
    clearable: true,
}));
const __VLS_47 = __VLS_46({
    modelValue: (__VLS_ctx.listQuery.keyword),
    ...{ class: "input-width" },
    placeholder: "帐号/姓名",
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
    ...{ class: "btn-add" },
}));
const __VLS_69 = __VLS_68({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
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
    ref: "adminTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_77 = __VLS_76({
    ref: "adminTable",
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
    label: "帐号",
    align: "center",
}));
const __VLS_92 = __VLS_91({
    label: "帐号",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_91));
const { default: __VLS_95 } = __VLS_93.slots;
{
    const { default: __VLS_96 } = __VLS_93.slots;
    const [scope] = __VLS_vSlot(__VLS_96);
    (scope.row.username);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_93;
let __VLS_97;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_98 = __VLS_asFunctionalComponent1(__VLS_97, new __VLS_97({
    label: "姓名",
    align: "center",
}));
const __VLS_99 = __VLS_98({
    label: "姓名",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_98));
const { default: __VLS_102 } = __VLS_100.slots;
{
    const { default: __VLS_103 } = __VLS_100.slots;
    const [scope] = __VLS_vSlot(__VLS_103);
    (scope.row.nickName);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_100;
let __VLS_104;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_105 = __VLS_asFunctionalComponent1(__VLS_104, new __VLS_104({
    label: "邮箱",
    align: "center",
}));
const __VLS_106 = __VLS_105({
    label: "邮箱",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_105));
const { default: __VLS_109 } = __VLS_107.slots;
{
    const { default: __VLS_110 } = __VLS_107.slots;
    const [scope] = __VLS_vSlot(__VLS_110);
    (scope.row.email);
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
    label: "最后登录",
    width: "160",
    align: "center",
}));
const __VLS_120 = __VLS_119({
    label: "最后登录",
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_119));
const { default: __VLS_123 } = __VLS_121.slots;
{
    const { default: __VLS_124 } = __VLS_121.slots;
    const [scope] = __VLS_vSlot(__VLS_124);
    (__VLS_ctx.formatDateTime(scope.row.loginTime));
    // @ts-ignore
    [formatDateTime,];
}
// @ts-ignore
[];
var __VLS_121;
let __VLS_125;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_126 = __VLS_asFunctionalComponent1(__VLS_125, new __VLS_125({
    label: "是否启用",
    width: "140",
    align: "center",
}));
const __VLS_127 = __VLS_126({
    label: "是否启用",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_126));
const { default: __VLS_130 } = __VLS_128.slots;
{
    const { default: __VLS_131 } = __VLS_128.slots;
    const [scope] = __VLS_vSlot(__VLS_131);
    let __VLS_132;
    /** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
    elSwitch;
    // @ts-ignore
    const __VLS_133 = __VLS_asFunctionalComponent1(__VLS_132, new __VLS_132({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.status),
    }));
    const __VLS_134 = __VLS_133({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.status),
    }, ...__VLS_functionalComponentArgsRest(__VLS_133));
    let __VLS_137;
    const __VLS_138 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handleStatusChange(scope.$index, scope.row);
                // @ts-ignore
                [handleStatusChange,];
            } });
    var __VLS_135;
    var __VLS_136;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_128;
let __VLS_139;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_140 = __VLS_asFunctionalComponent1(__VLS_139, new __VLS_139({
    label: "操作",
    width: "180",
    align: "center",
}));
const __VLS_141 = __VLS_140({
    label: "操作",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_140));
const { default: __VLS_144 } = __VLS_142.slots;
{
    const { default: __VLS_145 } = __VLS_142.slots;
    const [scope] = __VLS_vSlot(__VLS_145);
    let __VLS_146;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_147 = __VLS_asFunctionalComponent1(__VLS_146, new __VLS_146({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_148 = __VLS_147({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_147));
    let __VLS_151;
    const __VLS_152 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleSelectRole(scope.$index, scope.row);
                // @ts-ignore
                [handleSelectRole,];
            } });
    const { default: __VLS_153 } = __VLS_149.slots;
    // @ts-ignore
    [];
    var __VLS_149;
    var __VLS_150;
    let __VLS_154;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_155 = __VLS_asFunctionalComponent1(__VLS_154, new __VLS_154({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_156 = __VLS_155({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_155));
    let __VLS_159;
    const __VLS_160 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdate(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdate,];
            } });
    const { default: __VLS_161 } = __VLS_157.slots;
    // @ts-ignore
    [];
    var __VLS_157;
    var __VLS_158;
    let __VLS_162;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_163 = __VLS_asFunctionalComponent1(__VLS_162, new __VLS_162({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_164 = __VLS_163({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_163));
    let __VLS_167;
    const __VLS_168 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_169 } = __VLS_165.slots;
    // @ts-ignore
    [];
    var __VLS_165;
    var __VLS_166;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_142;
// @ts-ignore
[];
var __VLS_78;
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
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}));
const __VLS_172 = __VLS_171({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_171));
let __VLS_175;
const __VLS_176 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_177 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_173;
var __VLS_174;
let __VLS_178;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_179 = __VLS_asFunctionalComponent1(__VLS_178, new __VLS_178({
    modelValue: (__VLS_ctx.dialogVisible),
    title: (__VLS_ctx.isEdit ? '编辑用户' : '添加用户'),
    width: "40%",
}));
const __VLS_180 = __VLS_179({
    modelValue: (__VLS_ctx.dialogVisible),
    title: (__VLS_ctx.isEdit ? '编辑用户' : '添加用户'),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_179));
const { default: __VLS_183 } = __VLS_181.slots;
let __VLS_184;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_185 = __VLS_asFunctionalComponent1(__VLS_184, new __VLS_184({
    model: (__VLS_ctx.admin),
    labelWidth: "150px",
    size: "default",
}));
const __VLS_186 = __VLS_185({
    model: (__VLS_ctx.admin),
    labelWidth: "150px",
    size: "default",
}, ...__VLS_functionalComponentArgsRest(__VLS_185));
const { default: __VLS_189 } = __VLS_187.slots;
let __VLS_190;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_191 = __VLS_asFunctionalComponent1(__VLS_190, new __VLS_190({
    label: "帐号：",
}));
const __VLS_192 = __VLS_191({
    label: "帐号：",
}, ...__VLS_functionalComponentArgsRest(__VLS_191));
const { default: __VLS_195 } = __VLS_193.slots;
let __VLS_196;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_197 = __VLS_asFunctionalComponent1(__VLS_196, new __VLS_196({
    modelValue: (__VLS_ctx.admin.username),
    ...{ style: {} },
}));
const __VLS_198 = __VLS_197({
    modelValue: (__VLS_ctx.admin.username),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_197));
// @ts-ignore
[listQuery, listQuery, total, handleSizeChange, handleCurrentChange, dialogVisible, isEdit, admin, admin,];
var __VLS_193;
let __VLS_201;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_202 = __VLS_asFunctionalComponent1(__VLS_201, new __VLS_201({
    label: "姓名：",
}));
const __VLS_203 = __VLS_202({
    label: "姓名：",
}, ...__VLS_functionalComponentArgsRest(__VLS_202));
const { default: __VLS_206 } = __VLS_204.slots;
let __VLS_207;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_208 = __VLS_asFunctionalComponent1(__VLS_207, new __VLS_207({
    modelValue: (__VLS_ctx.admin.nickName),
    ...{ style: {} },
}));
const __VLS_209 = __VLS_208({
    modelValue: (__VLS_ctx.admin.nickName),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_208));
// @ts-ignore
[admin,];
var __VLS_204;
let __VLS_212;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_213 = __VLS_asFunctionalComponent1(__VLS_212, new __VLS_212({
    label: "邮箱：",
}));
const __VLS_214 = __VLS_213({
    label: "邮箱：",
}, ...__VLS_functionalComponentArgsRest(__VLS_213));
const { default: __VLS_217 } = __VLS_215.slots;
let __VLS_218;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_219 = __VLS_asFunctionalComponent1(__VLS_218, new __VLS_218({
    modelValue: (__VLS_ctx.admin.email),
    ...{ style: {} },
}));
const __VLS_220 = __VLS_219({
    modelValue: (__VLS_ctx.admin.email),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_219));
// @ts-ignore
[admin,];
var __VLS_215;
if (!__VLS_ctx.isEdit) {
    let __VLS_223;
    /** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
    elFormItem;
    // @ts-ignore
    const __VLS_224 = __VLS_asFunctionalComponent1(__VLS_223, new __VLS_223({
        label: "密码：",
    }));
    const __VLS_225 = __VLS_224({
        label: "密码：",
    }, ...__VLS_functionalComponentArgsRest(__VLS_224));
    const { default: __VLS_228 } = __VLS_226.slots;
    let __VLS_229;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_230 = __VLS_asFunctionalComponent1(__VLS_229, new __VLS_229({
        modelValue: (__VLS_ctx.admin.password),
        type: "password",
        ...{ style: {} },
    }));
    const __VLS_231 = __VLS_230({
        modelValue: (__VLS_ctx.admin.password),
        type: "password",
        ...{ style: {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_230));
    // @ts-ignore
    [isEdit, admin,];
    var __VLS_226;
}
let __VLS_234;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_235 = __VLS_asFunctionalComponent1(__VLS_234, new __VLS_234({
    label: "备注：",
}));
const __VLS_236 = __VLS_235({
    label: "备注：",
}, ...__VLS_functionalComponentArgsRest(__VLS_235));
const { default: __VLS_239 } = __VLS_237.slots;
let __VLS_240;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_241 = __VLS_asFunctionalComponent1(__VLS_240, new __VLS_240({
    modelValue: (__VLS_ctx.admin.note),
    type: "textarea",
    rows: (5),
    ...{ style: {} },
}));
const __VLS_242 = __VLS_241({
    modelValue: (__VLS_ctx.admin.note),
    type: "textarea",
    rows: (5),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_241));
// @ts-ignore
[admin,];
var __VLS_237;
let __VLS_245;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_246 = __VLS_asFunctionalComponent1(__VLS_245, new __VLS_245({
    label: "是否启用：",
}));
const __VLS_247 = __VLS_246({
    label: "是否启用：",
}, ...__VLS_functionalComponentArgsRest(__VLS_246));
const { default: __VLS_250 } = __VLS_248.slots;
let __VLS_251;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_252 = __VLS_asFunctionalComponent1(__VLS_251, new __VLS_251({
    modelValue: (__VLS_ctx.admin.status),
}));
const __VLS_253 = __VLS_252({
    modelValue: (__VLS_ctx.admin.status),
}, ...__VLS_functionalComponentArgsRest(__VLS_252));
const { default: __VLS_256 } = __VLS_254.slots;
let __VLS_257;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_258 = __VLS_asFunctionalComponent1(__VLS_257, new __VLS_257({
    label: (1),
}));
const __VLS_259 = __VLS_258({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_258));
const { default: __VLS_262 } = __VLS_260.slots;
// @ts-ignore
[admin,];
var __VLS_260;
let __VLS_263;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_264 = __VLS_asFunctionalComponent1(__VLS_263, new __VLS_263({
    label: (0),
}));
const __VLS_265 = __VLS_264({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_264));
const { default: __VLS_268 } = __VLS_266.slots;
// @ts-ignore
[];
var __VLS_266;
// @ts-ignore
[];
var __VLS_254;
// @ts-ignore
[];
var __VLS_248;
// @ts-ignore
[];
var __VLS_187;
{
    const { footer: __VLS_269 } = __VLS_181.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_270;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_271 = __VLS_asFunctionalComponent1(__VLS_270, new __VLS_270({
        ...{ 'onClick': {} },
        size: "default",
    }));
    const __VLS_272 = __VLS_271({
        ...{ 'onClick': {} },
        size: "default",
    }, ...__VLS_functionalComponentArgsRest(__VLS_271));
    let __VLS_275;
    const __VLS_276 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.dialogVisible = false;
                // @ts-ignore
                [dialogVisible,];
            } });
    const { default: __VLS_277 } = __VLS_273.slots;
    // @ts-ignore
    [];
    var __VLS_273;
    var __VLS_274;
    let __VLS_278;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_279 = __VLS_asFunctionalComponent1(__VLS_278, new __VLS_278({
        ...{ 'onClick': {} },
        type: "primary",
        size: "default",
    }));
    const __VLS_280 = __VLS_279({
        ...{ 'onClick': {} },
        type: "primary",
        size: "default",
    }, ...__VLS_functionalComponentArgsRest(__VLS_279));
    let __VLS_283;
    const __VLS_284 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDialogConfirm();
                // @ts-ignore
                [handleDialogConfirm,];
            } });
    const { default: __VLS_285 } = __VLS_281.slots;
    // @ts-ignore
    [];
    var __VLS_281;
    var __VLS_282;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_181;
let __VLS_286;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_287 = __VLS_asFunctionalComponent1(__VLS_286, new __VLS_286({
    title: "分配角色",
    modelValue: (__VLS_ctx.allocDialogVisible),
    width: "30%",
}));
const __VLS_288 = __VLS_287({
    title: "分配角色",
    modelValue: (__VLS_ctx.allocDialogVisible),
    width: "30%",
}, ...__VLS_functionalComponentArgsRest(__VLS_287));
const { default: __VLS_291 } = __VLS_289.slots;
let __VLS_292;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_293 = __VLS_asFunctionalComponent1(__VLS_292, new __VLS_292({
    modelValue: (__VLS_ctx.allocRoleIds),
    multiple: true,
    placeholder: "请选择",
    size: "default",
    ...{ style: {} },
}));
const __VLS_294 = __VLS_293({
    modelValue: (__VLS_ctx.allocRoleIds),
    multiple: true,
    placeholder: "请选择",
    size: "default",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_293));
const { default: __VLS_297 } = __VLS_295.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.allRoleList))) {
    let __VLS_298;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_299 = __VLS_asFunctionalComponent1(__VLS_298, new __VLS_298({
        key: (item.id),
        label: (item.name),
        value: (item.id),
    }));
    const __VLS_300 = __VLS_299({
        key: (item.id),
        label: (item.name),
        value: (item.id),
    }, ...__VLS_functionalComponentArgsRest(__VLS_299));
    // @ts-ignore
    [allocDialogVisible, allocRoleIds, allRoleList,];
}
// @ts-ignore
[];
var __VLS_295;
{
    const { footer: __VLS_303 } = __VLS_289.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_304;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_305 = __VLS_asFunctionalComponent1(__VLS_304, new __VLS_304({
        ...{ 'onClick': {} },
        size: "default",
    }));
    const __VLS_306 = __VLS_305({
        ...{ 'onClick': {} },
        size: "default",
    }, ...__VLS_functionalComponentArgsRest(__VLS_305));
    let __VLS_309;
    const __VLS_310 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.allocDialogVisible = false;
                // @ts-ignore
                [allocDialogVisible,];
            } });
    const { default: __VLS_311 } = __VLS_307.slots;
    // @ts-ignore
    [];
    var __VLS_307;
    var __VLS_308;
    let __VLS_312;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_313 = __VLS_asFunctionalComponent1(__VLS_312, new __VLS_312({
        ...{ 'onClick': {} },
        type: "primary",
        size: "default",
    }));
    const __VLS_314 = __VLS_313({
        ...{ 'onClick': {} },
        type: "primary",
        size: "default",
    }, ...__VLS_functionalComponentArgsRest(__VLS_313));
    let __VLS_317;
    const __VLS_318 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleAllocDialogConfirm();
                // @ts-ignore
                [handleAllocDialogConfirm,];
            } });
    const { default: __VLS_319 } = __VLS_315.slots;
    // @ts-ignore
    [];
    var __VLS_315;
    var __VLS_316;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_289;
// @ts-ignore
var __VLS_81 = __VLS_80;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
