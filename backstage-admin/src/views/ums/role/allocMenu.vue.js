/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getMenuTreeListAPI } from '@/apis/menu.ts'; // 修改路径为apis
import { roleListMenuByRoleIdAPI, roleAllocMenuAPI } from '@/apis/role'; // 修改路径为apis
// 获取路由参数
const router = useRouter();
const route = useRoute();
// 当前角色ID
const roleId = ref();
// 所有菜单树形结构列表
const menuTreeList = ref([]);
// 创建对树组件的引用
const treeRef = ref();
// 定义树组件属性
const defaultProps = {
    children: 'children',
    label: 'title'
};
// 获取菜单树列表
const treeList = async () => {
    const res = await getMenuTreeListAPI();
    menuTreeList.value = res.data;
};
// 获取角色对应的菜单
const getRoleMenu = async () => {
    const res = await roleListMenuByRoleIdAPI(roleId.value);
    const menuList = res.data;
    const checkedMenuIds = menuList.filter(item => item.parentId !== 0).map(item => item.id);
    treeRef.value.setCheckedKeys(checkedMenuIds);
};
// 页面创建时执行
onMounted(() => {
    roleId.value = Number(route.query.roleId);
    treeList();
    getRoleMenu();
});
// 保存菜单分配
const handleSave = async () => {
    const checkedNodes = treeRef.value.getCheckedNodes();
    const checkedMenuIds = new Set();
    checkedNodes.forEach(item => {
        checkedMenuIds.add(item.id);
        if (item.parentId !== 0) {
            checkedMenuIds.add(item.parentId);
        }
    });
    await ElMessageBox.confirm('是否分配菜单？', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    await roleAllocMenuAPI({ roleId: roleId.value, menuIds: Array.from(checkedMenuIds).join(',') });
    ElMessage({
        message: '分配成功',
        type: 'success',
        duration: 1000
    });
    router.back();
};
// 清空选中项
const handleClear = () => {
    treeRef.value.setCheckedKeys([]);
};
const __VLS_ctx = {
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    ...{ class: "form-container" },
    shadow: "never",
}));
const __VLS_2 = __VLS_1({
    ...{ class: "form-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_5 = {};
/** @type {__VLS_StyleScopedClasses['form-container']} */ ;
const { default: __VLS_6 } = __VLS_3.slots;
let __VLS_7;
/** @ts-ignore @type { | typeof __VLS_components.elTree | typeof __VLS_components.ElTree | typeof __VLS_components['el-tree'] | typeof __VLS_components.elTree | typeof __VLS_components.ElTree | typeof __VLS_components['el-tree']} */
elTree;
// @ts-ignore
const __VLS_8 = __VLS_asFunctionalComponent1(__VLS_7, new __VLS_7({
    ref: "treeRef",
    data: (__VLS_ctx.menuTreeList),
    showCheckbox: true,
    defaultExpandAll: true,
    nodeKey: "id",
    props: (__VLS_ctx.defaultProps),
    highlightCurrent: true,
}));
const __VLS_9 = __VLS_8({
    ref: "treeRef",
    data: (__VLS_ctx.menuTreeList),
    showCheckbox: true,
    defaultExpandAll: true,
    nodeKey: "id",
    props: (__VLS_ctx.defaultProps),
    highlightCurrent: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
var __VLS_12 = {};
var __VLS_10;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
    align: "center",
});
let __VLS_14;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_15 = __VLS_asFunctionalComponent1(__VLS_14, new __VLS_14({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_16 = __VLS_15({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
let __VLS_19;
const __VLS_20 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleSave();
            // @ts-ignore
            [menuTreeList, defaultProps, handleSave,];
        } });
const { default: __VLS_21 } = __VLS_17.slots;
// @ts-ignore
[];
var __VLS_17;
var __VLS_18;
let __VLS_22;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_23 = __VLS_asFunctionalComponent1(__VLS_22, new __VLS_22({
    ...{ 'onClick': {} },
}));
const __VLS_24 = __VLS_23({
    ...{ 'onClick': {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_23));
let __VLS_27;
const __VLS_28 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleClear();
            // @ts-ignore
            [handleClear,];
        } });
const { default: __VLS_29 } = __VLS_25.slots;
// @ts-ignore
[];
var __VLS_25;
var __VLS_26;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
var __VLS_13 = __VLS_12;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
