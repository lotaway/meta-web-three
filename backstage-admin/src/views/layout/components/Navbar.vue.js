/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { computed } from 'vue';
import Breadcrumb from '@/components/Breadcrumb/index.vue';
import Hamburger from '@/components/Hamburger/index.vue';
import { useAppStore } from '@/stores/app';
import { useUserStore } from '@/stores/user';
// 定义组件名称
defineOptions({
    name: 'Navbar'
});
const appStore = useAppStore();
const userStore = useUserStore();
const sidebar = computed(() => appStore.sidebar);
const avatar = computed(() => userStore.userInfo.avatar);
// 处理开关侧边栏操作
const handleToggleSideBar = () => {
    appStore.toggleSideBar();
};
// 处理用户登出
const handleLogout = async () => {
    await userStore.userLogout();
    // 为了重新实例化vue-router对象 避免bug
    location.reload();
};
const __VLS_ctx = {
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elMenu | typeof __VLS_components.ElMenu | typeof __VLS_components['el-menu'] | typeof __VLS_components.elMenu | typeof __VLS_components.ElMenu | typeof __VLS_components['el-menu']} */
elMenu;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    ...{ class: "navbar" },
    mode: "horizontal",
}));
const __VLS_2 = __VLS_1({
    ...{ class: "navbar" },
    mode: "horizontal",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_5 = {};
/** @type {__VLS_StyleScopedClasses['navbar']} */ ;
const { default: __VLS_6 } = __VLS_3.slots;
const __VLS_7 = Hamburger || Hamburger;
// @ts-ignore
const __VLS_8 = __VLS_asFunctionalComponent1(__VLS_7, new __VLS_7({
    ...{ class: "hamburger-container" },
    toggleClick: (__VLS_ctx.handleToggleSideBar),
    isActive: (__VLS_ctx.sidebar.opened),
}));
const __VLS_9 = __VLS_8({
    ...{ class: "hamburger-container" },
    toggleClick: (__VLS_ctx.handleToggleSideBar),
    isActive: (__VLS_ctx.sidebar.opened),
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
/** @type {__VLS_StyleScopedClasses['hamburger-container']} */ ;
const __VLS_12 = Breadcrumb || Breadcrumb;
// @ts-ignore
const __VLS_13 = __VLS_asFunctionalComponent1(__VLS_12, new __VLS_12({}));
const __VLS_14 = __VLS_13({}, ...__VLS_functionalComponentArgsRest(__VLS_13));
let __VLS_17;
/** @ts-ignore @type { | typeof __VLS_components.elDropdown | typeof __VLS_components.ElDropdown | typeof __VLS_components['el-dropdown'] | typeof __VLS_components.elDropdown | typeof __VLS_components.ElDropdown | typeof __VLS_components['el-dropdown']} */
elDropdown;
// @ts-ignore
const __VLS_18 = __VLS_asFunctionalComponent1(__VLS_17, new __VLS_17({
    ...{ class: "avatar-container" },
    trigger: "click",
}));
const __VLS_19 = __VLS_18({
    ...{ class: "avatar-container" },
    trigger: "click",
}, ...__VLS_functionalComponentArgsRest(__VLS_18));
/** @type {__VLS_StyleScopedClasses['avatar-container']} */ ;
const { default: __VLS_22 } = __VLS_20.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "avatar-wrapper" },
});
/** @type {__VLS_StyleScopedClasses['avatar-wrapper']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.img)({
    ...{ class: "user-avatar" },
    src: (__VLS_ctx.avatar),
});
/** @type {__VLS_StyleScopedClasses['user-avatar']} */ ;
let __VLS_23;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_24 = __VLS_asFunctionalComponent1(__VLS_23, new __VLS_23({
    ...{ class: "el-icon-caret-bottom" },
}));
const __VLS_25 = __VLS_24({
    ...{ class: "el-icon-caret-bottom" },
}, ...__VLS_functionalComponentArgsRest(__VLS_24));
/** @type {__VLS_StyleScopedClasses['el-icon-caret-bottom']} */ ;
const { default: __VLS_28 } = __VLS_26.slots;
let __VLS_29;
/** @ts-ignore @type { | typeof __VLS_components.arrowDown | typeof __VLS_components.ArrowDown | typeof __VLS_components['arrow-down']} */
arrowDown;
// @ts-ignore
const __VLS_30 = __VLS_asFunctionalComponent1(__VLS_29, new __VLS_29({}));
const __VLS_31 = __VLS_30({}, ...__VLS_functionalComponentArgsRest(__VLS_30));
// @ts-ignore
[handleToggleSideBar, sidebar, avatar,];
var __VLS_26;
{
    const { dropdown: __VLS_34 } = __VLS_20.slots;
    let __VLS_35;
    /** @ts-ignore @type { | typeof __VLS_components.elDropdownMenu | typeof __VLS_components.ElDropdownMenu | typeof __VLS_components['el-dropdown-menu'] | typeof __VLS_components.elDropdownMenu | typeof __VLS_components.ElDropdownMenu | typeof __VLS_components['el-dropdown-menu']} */
    elDropdownMenu;
    // @ts-ignore
    const __VLS_36 = __VLS_asFunctionalComponent1(__VLS_35, new __VLS_35({
        ...{ class: "user-dropdown" },
    }));
    const __VLS_37 = __VLS_36({
        ...{ class: "user-dropdown" },
    }, ...__VLS_functionalComponentArgsRest(__VLS_36));
    /** @type {__VLS_StyleScopedClasses['user-dropdown']} */ ;
    const { default: __VLS_40 } = __VLS_38.slots;
    let __VLS_41;
    /** @ts-ignore @type { | typeof __VLS_components.routerLink | typeof __VLS_components.RouterLink | typeof __VLS_components['router-link'] | typeof __VLS_components.routerLink | typeof __VLS_components.RouterLink | typeof __VLS_components['router-link']} */
    routerLink;
    // @ts-ignore
    const __VLS_42 = __VLS_asFunctionalComponent1(__VLS_41, new __VLS_41({
        ...{ class: "inlineBlock" },
        to: "/",
    }));
    const __VLS_43 = __VLS_42({
        ...{ class: "inlineBlock" },
        to: "/",
    }, ...__VLS_functionalComponentArgsRest(__VLS_42));
    /** @type {__VLS_StyleScopedClasses['inlineBlock']} */ ;
    const { default: __VLS_46 } = __VLS_44.slots;
    let __VLS_47;
    /** @ts-ignore @type { | typeof __VLS_components.elDropdownItem | typeof __VLS_components.ElDropdownItem | typeof __VLS_components['el-dropdown-item'] | typeof __VLS_components.elDropdownItem | typeof __VLS_components.ElDropdownItem | typeof __VLS_components['el-dropdown-item']} */
    elDropdownItem;
    // @ts-ignore
    const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({}));
    const __VLS_49 = __VLS_48({}, ...__VLS_functionalComponentArgsRest(__VLS_48));
    const { default: __VLS_52 } = __VLS_50.slots;
    // @ts-ignore
    [];
    var __VLS_50;
    // @ts-ignore
    [];
    var __VLS_44;
    let __VLS_53;
    /** @ts-ignore @type { | typeof __VLS_components.elDropdownItem | typeof __VLS_components.ElDropdownItem | typeof __VLS_components['el-dropdown-item'] | typeof __VLS_components.elDropdownItem | typeof __VLS_components.ElDropdownItem | typeof __VLS_components['el-dropdown-item']} */
    elDropdownItem;
    // @ts-ignore
    const __VLS_54 = __VLS_asFunctionalComponent1(__VLS_53, new __VLS_53({
        divided: true,
    }));
    const __VLS_55 = __VLS_54({
        divided: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_54));
    const { default: __VLS_58 } = __VLS_56.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ onClick: (__VLS_ctx.handleLogout) },
        ...{ style: {} },
    });
    // @ts-ignore
    [handleLogout,];
    var __VLS_56;
    // @ts-ignore
    [];
    var __VLS_38;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_20;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
