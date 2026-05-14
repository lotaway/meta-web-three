/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { fetchAllResourceList } from '@/apis/resource.ts';
import { resourceCategoryListAllAPI } from '@/apis/resourceCategory.ts';
import { roleAllocResourceAPI, roleListResourceById } from '@/apis/role';
import { ElMessage, ElMessageBox } from 'element-plus';
// 获取路由
const router = useRouter();
const route = useRoute();
// 当前操作的角色ID
const roleId = ref();
// 所有资源列表
const allResource = ref([]);
// 所有资源分类列表
const allResourceCate = ref([]);
// 获取所有资源分类列表
const getAllResourceCateList = async () => {
    const res = await resourceCategoryListAllAPI();
    allResourceCate.value = res.data;
    allResourceCate.value.forEach(item => item.checked = false);
};
// 获取所有资源列表
const getAllResourceList = async () => {
    const res = await fetchAllResourceList();
    allResource.value = res.data;
    allResource.value.forEach(item => item.checked = false);
};
// 根据角色获取已分配资源并设置选中状态
const getResourceByRole = async () => {
    const res = await roleListResourceById(roleId.value);
    const allocResource = res.data;
    allResource.value.forEach(item => {
        item.checked = getResourceChecked(item.id, allocResource);
    });
    allResourceCate.value.forEach(item => {
        item.checked = isAllChecked(item.id);
    });
};
// 页面挂载时执行
onMounted(async () => {
    roleId.value = Number(route.query.roleId);
    await getAllResourceCateList();
    await getAllResourceList();
    await getResourceByRole();
});
// 根据分类ID获取资源
const getResourceByCate = (categoryId) => {
    return allResource.value.filter(item => item.categoryId === categoryId);
};
// 检查资源是否被选中
const getResourceChecked = (resourceId, allocResource) => {
    const index = allocResource.findIndex(item => item.id === resourceId);
    return index > -1;
};
// 检查分类是否半选状态
const isIndeterminate = (categoryId) => {
    const cateResources = getResourceByCate(categoryId);
    return !(cateResources.every(item => item.checked === true) || cateResources.every(item => item.checked === false));
};
// 检查分类是否全选
const isAllChecked = (categoryId) => {
    const cateResources = getResourceByCate(categoryId);
    return cateResources.every(item => item.checked === true);
};
// 保存资源分配
const handleSave = async () => {
    await ElMessageBox.confirm('是否分配资源？', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    const checkedResourceIds = new Set();
    if (allResource.value && allResource.value.length > 0) {
        allResource.value.forEach(item => {
            if (item.checked) {
                checkedResourceIds.add(item.id);
            }
        });
    }
    await roleAllocResourceAPI({ roleId: roleId.value, resourceIds: Array.from(checkedResourceIds).join(',') });
    ElMessage({
        message: '分配成功',
        type: 'success',
        duration: 1000
    });
    router.back();
};
// 清空选中项
const handleClear = () => {
    allResourceCate.value.forEach(item => {
        item.checked = false;
    });
    allResource.value.forEach(item => {
        item.checked = false;
    });
};
// 处理全选改变事件
const handleCheckAllChange = (cate) => {
    const cateResources = getResourceByCate(cate.id);
    cateResources.forEach(item => item.checked = cate.checked);
};
// 处理单个资源选中事件
const handleCheckChange = (resource) => {
    allResourceCate.value.forEach(item => {
        if (item.id === resource.categoryId) {
            item.checked = isAllChecked(resource.categoryId);
        }
    });
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
for (const [cate, index] of __VLS_vFor((__VLS_ctx.allResourceCate))) {
    __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
        ...{ class: (index === 0 ? 'top-line' : null) },
        key: ('cate' + cate.id),
    });
    let __VLS_7;
    /** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
    elRow;
    // @ts-ignore
    const __VLS_8 = __VLS_asFunctionalComponent1(__VLS_7, new __VLS_7({
        ...{ class: "table-layout" },
        ...{ style: {} },
    }));
    const __VLS_9 = __VLS_8({
        ...{ class: "table-layout" },
        ...{ style: {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_8));
    /** @type {__VLS_StyleScopedClasses['table-layout']} */ ;
    const { default: __VLS_12 } = __VLS_10.slots;
    let __VLS_13;
    /** @ts-ignore @type { | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox'] | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox']} */
    elCheckbox;
    // @ts-ignore
    const __VLS_14 = __VLS_asFunctionalComponent1(__VLS_13, new __VLS_13({
        ...{ 'onChange': {} },
        modelValue: (cate.checked),
        indeterminate: (__VLS_ctx.isIndeterminate(cate.id)),
    }));
    const __VLS_15 = __VLS_14({
        ...{ 'onChange': {} },
        modelValue: (cate.checked),
        indeterminate: (__VLS_ctx.isIndeterminate(cate.id)),
    }, ...__VLS_functionalComponentArgsRest(__VLS_14));
    let __VLS_18;
    const __VLS_19 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handleCheckAllChange(cate);
                // @ts-ignore
                [allResourceCate, isIndeterminate, handleCheckAllChange,];
            } });
    const { default: __VLS_20 } = __VLS_16.slots;
    (cate.name);
    // @ts-ignore
    [];
    var __VLS_16;
    var __VLS_17;
    // @ts-ignore
    [];
    var __VLS_10;
    let __VLS_21;
    /** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
    elRow;
    // @ts-ignore
    const __VLS_22 = __VLS_asFunctionalComponent1(__VLS_21, new __VLS_21({
        ...{ class: "table-layout" },
    }));
    const __VLS_23 = __VLS_22({
        ...{ class: "table-layout" },
    }, ...__VLS_functionalComponentArgsRest(__VLS_22));
    /** @type {__VLS_StyleScopedClasses['table-layout']} */ ;
    const { default: __VLS_26 } = __VLS_24.slots;
    for (const [resource] of __VLS_vFor((__VLS_ctx.getResourceByCate(cate.id)))) {
        let __VLS_27;
        /** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
        elCol;
        // @ts-ignore
        const __VLS_28 = __VLS_asFunctionalComponent1(__VLS_27, new __VLS_27({
            span: (8),
            key: (resource.id),
            ...{ style: {} },
        }));
        const __VLS_29 = __VLS_28({
            span: (8),
            key: (resource.id),
            ...{ style: {} },
        }, ...__VLS_functionalComponentArgsRest(__VLS_28));
        const { default: __VLS_32 } = __VLS_30.slots;
        let __VLS_33;
        /** @ts-ignore @type { | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox'] | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox']} */
        elCheckbox;
        // @ts-ignore
        const __VLS_34 = __VLS_asFunctionalComponent1(__VLS_33, new __VLS_33({
            ...{ 'onChange': {} },
            modelValue: (resource.checked),
        }));
        const __VLS_35 = __VLS_34({
            ...{ 'onChange': {} },
            modelValue: (resource.checked),
        }, ...__VLS_functionalComponentArgsRest(__VLS_34));
        let __VLS_38;
        const __VLS_39 = ({ change: {} },
            { onChange: (...[$event]) => {
                    __VLS_ctx.handleCheckChange(resource);
                    // @ts-ignore
                    [getResourceByCate, handleCheckChange,];
                } });
        const { default: __VLS_40 } = __VLS_36.slots;
        (resource.name);
        // @ts-ignore
        [];
        var __VLS_36;
        var __VLS_37;
        // @ts-ignore
        [];
        var __VLS_30;
        // @ts-ignore
        [];
    }
    // @ts-ignore
    [];
    var __VLS_24;
    // @ts-ignore
    [];
}
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
    align: "center",
});
let __VLS_41;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_42 = __VLS_asFunctionalComponent1(__VLS_41, new __VLS_41({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_43 = __VLS_42({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_42));
let __VLS_46;
const __VLS_47 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleSave();
            // @ts-ignore
            [handleSave,];
        } });
const { default: __VLS_48 } = __VLS_44.slots;
// @ts-ignore
[];
var __VLS_44;
var __VLS_45;
let __VLS_49;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_50 = __VLS_asFunctionalComponent1(__VLS_49, new __VLS_49({
    ...{ 'onClick': {} },
}));
const __VLS_51 = __VLS_50({
    ...{ 'onClick': {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_50));
let __VLS_54;
const __VLS_55 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleClear();
            // @ts-ignore
            [handleClear,];
        } });
const { default: __VLS_56 } = __VLS_52.slots;
// @ts-ignore
[];
var __VLS_52;
var __VLS_53;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
