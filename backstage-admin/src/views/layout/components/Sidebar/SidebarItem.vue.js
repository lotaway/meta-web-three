/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { computed } from 'vue';
// 定义组件名称
defineOptions({
    name: 'SidebarItem'
});
// 定义props
const props = defineProps({
    // 生成菜单的路由
    routes: {
        type: Array
    },
    // 控制只有一个子菜单的一级菜单样式
    isNest: {
        type: Boolean,
        default: false
    }
});
// 过滤出需要显示的路由
const filteredRoutes = computed(() => {
    return props.routes.filter(item => !item.hidden && item.children);
});
// 过滤出需要显示的子路由
const getFilteredChildren = (children) => {
    return children.filter(child => !child.hidden);
};
// 判断路由下方是否只有一个子路由
const hasOneShowingChildren = (children) => {
    const showingChildren = children.filter(item => {
        return !item.hidden;
    });
    if (showingChildren.length === 1) {
        return true;
    }
    return false;
};
const __VLS_ctx = {
    ...{},
    ...{},
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "menu-wrapper" },
});
/** @type {__VLS_StyleScopedClasses['menu-wrapper']} */ ;
for (const [item] of __VLS_vFor((__VLS_ctx.filteredRoutes))) {
    if (item.children && __VLS_ctx.hasOneShowingChildren(item.children) && !item.children[0].children && !item.alwaysShow) {
        let __VLS_0;
        /** @ts-ignore @type { | typeof __VLS_components.routerLink | typeof __VLS_components.RouterLink | typeof __VLS_components['router-link'] | typeof __VLS_components.routerLink | typeof __VLS_components.RouterLink | typeof __VLS_components['router-link']} */
        routerLink;
        // @ts-ignore
        const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
            to: (item.path + '/' + item.children[0].path),
            key: (item.children[0].name),
        }));
        const __VLS_2 = __VLS_1({
            to: (item.path + '/' + item.children[0].path),
            key: (item.children[0].name),
        }, ...__VLS_functionalComponentArgsRest(__VLS_1));
        const { default: __VLS_5 } = __VLS_3.slots;
        let __VLS_6;
        /** @ts-ignore @type { | typeof __VLS_components.elMenuItem | typeof __VLS_components.ElMenuItem | typeof __VLS_components['el-menu-item'] | typeof __VLS_components.elMenuItem | typeof __VLS_components.ElMenuItem | typeof __VLS_components['el-menu-item']} */
        elMenuItem;
        // @ts-ignore
        const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({
            index: (item.path + '/' + item.children[0].path),
            ...{ class: ({ 'submenu-title-noDropdown': !__VLS_ctx.isNest }) },
        }));
        const __VLS_8 = __VLS_7({
            index: (item.path + '/' + item.children[0].path),
            ...{ class: ({ 'submenu-title-noDropdown': !__VLS_ctx.isNest }) },
        }, ...__VLS_functionalComponentArgsRest(__VLS_7));
        /** @type {__VLS_StyleScopedClasses['submenu-title-noDropdown']} */ ;
        const { default: __VLS_11 } = __VLS_9.slots;
        if (item.children[0].meta && item.children[0].meta.icon) {
            let __VLS_12;
            /** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
            svgIcon;
            // @ts-ignore
            const __VLS_13 = __VLS_asFunctionalComponent1(__VLS_12, new __VLS_12({
                iconClass: (item.children[0].meta.icon),
            }));
            const __VLS_14 = __VLS_13({
                iconClass: (item.children[0].meta.icon),
            }, ...__VLS_functionalComponentArgsRest(__VLS_13));
        }
        {
            const { title: __VLS_17 } = __VLS_9.slots;
            if (item.children[0].meta && item.children[0].meta.title) {
                __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
                (item.children[0].meta.title);
            }
            // @ts-ignore
            [filteredRoutes, hasOneShowingChildren, isNest,];
        }
        // @ts-ignore
        [];
        var __VLS_9;
        // @ts-ignore
        [];
        var __VLS_3;
    }
    else {
        let __VLS_18;
        /** @ts-ignore @type { | typeof __VLS_components.elSubMenu | typeof __VLS_components.ElSubMenu | typeof __VLS_components['el-sub-menu'] | typeof __VLS_components.elSubMenu | typeof __VLS_components.ElSubMenu | typeof __VLS_components['el-sub-menu']} */
        elSubMenu;
        // @ts-ignore
        const __VLS_19 = __VLS_asFunctionalComponent1(__VLS_18, new __VLS_18({
            index: (item.name || item.path),
            key: (item.name),
        }));
        const __VLS_20 = __VLS_19({
            index: (item.name || item.path),
            key: (item.name),
        }, ...__VLS_functionalComponentArgsRest(__VLS_19));
        const { default: __VLS_23 } = __VLS_21.slots;
        {
            const { title: __VLS_24 } = __VLS_21.slots;
            if (item.meta && item.meta.icon) {
                let __VLS_25;
                /** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
                svgIcon;
                // @ts-ignore
                const __VLS_26 = __VLS_asFunctionalComponent1(__VLS_25, new __VLS_25({
                    iconClass: (item.meta.icon),
                }));
                const __VLS_27 = __VLS_26({
                    iconClass: (item.meta.icon),
                }, ...__VLS_functionalComponentArgsRest(__VLS_26));
            }
            if (item.meta && item.meta.title) {
                __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
                (item.meta.title);
            }
            // @ts-ignore
            [];
        }
        for (const [child] of __VLS_vFor((__VLS_ctx.getFilteredChildren(item.children)))) {
            if (child.children && child.children.length > 0) {
                let __VLS_30;
                /** @ts-ignore @type { | typeof __VLS_components.sidebarItem | typeof __VLS_components.SidebarItem | typeof __VLS_components['sidebar-item'] | typeof __VLS_components.sidebarItem | typeof __VLS_components.SidebarItem | typeof __VLS_components['sidebar-item']} */
                sidebarItem;
                // @ts-ignore
                const __VLS_31 = __VLS_asFunctionalComponent1(__VLS_30, new __VLS_30({
                    isNest: (true),
                    ...{ class: "nest-menu" },
                    routes: ([child]),
                    key: (child.path),
                }));
                const __VLS_32 = __VLS_31({
                    isNest: (true),
                    ...{ class: "nest-menu" },
                    routes: ([child]),
                    key: (child.path),
                }, ...__VLS_functionalComponentArgsRest(__VLS_31));
                /** @type {__VLS_StyleScopedClasses['nest-menu']} */ ;
            }
            else if (child.path.startsWith('http')) {
                __VLS_asFunctionalElement1(__VLS_intrinsics.a, __VLS_intrinsics.a)({
                    href: (child.path),
                    target: "_blank",
                    key: (child.name),
                });
                let __VLS_35;
                /** @ts-ignore @type { | typeof __VLS_components.elMenuItem | typeof __VLS_components.ElMenuItem | typeof __VLS_components['el-menu-item'] | typeof __VLS_components.elMenuItem | typeof __VLS_components.ElMenuItem | typeof __VLS_components['el-menu-item']} */
                elMenuItem;
                // @ts-ignore
                const __VLS_36 = __VLS_asFunctionalComponent1(__VLS_35, new __VLS_35({
                    index: (item.path + '/' + child.path),
                }));
                const __VLS_37 = __VLS_36({
                    index: (item.path + '/' + child.path),
                }, ...__VLS_functionalComponentArgsRest(__VLS_36));
                const { default: __VLS_40 } = __VLS_38.slots;
                if (child.meta && child.meta.icon) {
                    let __VLS_41;
                    /** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
                    svgIcon;
                    // @ts-ignore
                    const __VLS_42 = __VLS_asFunctionalComponent1(__VLS_41, new __VLS_41({
                        iconClass: (child.meta.icon),
                    }));
                    const __VLS_43 = __VLS_42({
                        iconClass: (child.meta.icon),
                    }, ...__VLS_functionalComponentArgsRest(__VLS_42));
                }
                {
                    const { title: __VLS_46 } = __VLS_38.slots;
                    if (child.meta && child.meta.title) {
                        __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
                        (child.meta.title);
                    }
                    // @ts-ignore
                    [getFilteredChildren,];
                }
                // @ts-ignore
                [];
                var __VLS_38;
            }
            else {
                let __VLS_47;
                /** @ts-ignore @type { | typeof __VLS_components.routerLink | typeof __VLS_components.RouterLink | typeof __VLS_components['router-link'] | typeof __VLS_components.routerLink | typeof __VLS_components.RouterLink | typeof __VLS_components['router-link']} */
                routerLink;
                // @ts-ignore
                const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({
                    to: (item.path + '/' + child.path),
                    key: ('route-' + child.name),
                }));
                const __VLS_49 = __VLS_48({
                    to: (item.path + '/' + child.path),
                    key: ('route-' + child.name),
                }, ...__VLS_functionalComponentArgsRest(__VLS_48));
                const { default: __VLS_52 } = __VLS_50.slots;
                let __VLS_53;
                /** @ts-ignore @type { | typeof __VLS_components.elMenuItem | typeof __VLS_components.ElMenuItem | typeof __VLS_components['el-menu-item'] | typeof __VLS_components.elMenuItem | typeof __VLS_components.ElMenuItem | typeof __VLS_components['el-menu-item']} */
                elMenuItem;
                // @ts-ignore
                const __VLS_54 = __VLS_asFunctionalComponent1(__VLS_53, new __VLS_53({
                    index: (item.path + '/' + child.path),
                }));
                const __VLS_55 = __VLS_54({
                    index: (item.path + '/' + child.path),
                }, ...__VLS_functionalComponentArgsRest(__VLS_54));
                const { default: __VLS_58 } = __VLS_56.slots;
                if (child.meta && child.meta.icon) {
                    let __VLS_59;
                    /** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
                    svgIcon;
                    // @ts-ignore
                    const __VLS_60 = __VLS_asFunctionalComponent1(__VLS_59, new __VLS_59({
                        iconClass: (child.meta.icon),
                    }));
                    const __VLS_61 = __VLS_60({
                        iconClass: (child.meta.icon),
                    }, ...__VLS_functionalComponentArgsRest(__VLS_60));
                }
                {
                    const { title: __VLS_64 } = __VLS_56.slots;
                    if (child.meta && child.meta.title) {
                        __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
                        (child.meta.title);
                    }
                    // @ts-ignore
                    [];
                }
                // @ts-ignore
                [];
                var __VLS_56;
                // @ts-ignore
                [];
                var __VLS_50;
            }
            // @ts-ignore
            [];
        }
        // @ts-ignore
        [];
        var __VLS_21;
    }
    // @ts-ignore
    [];
}
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({
    props: {
        // 生成菜单的路由
        routes: {
            type: Array
        },
        // 控制只有一个子菜单的一级菜单样式
        isNest: {
            type: Boolean,
            default: false
        }
    },
});
export default {};
