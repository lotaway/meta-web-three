/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, reactive, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getMenuListByParentIdAPI, menuCreateAPI, updateMenu, getMenuByIdAPI } from '@/apis/menu.ts';
// 获取路由对象
const route = useRoute();
const router = useRouter();
// 定义属性
const props = defineProps({
    isEdit: {
        type: Boolean,
        default: false
    }
});
// 默认菜单对象
const defaultMenu = {
    title: '',
    parentId: 0,
    name: '',
    icon: '',
    hidden: 0,
    sort: 0
};
// 菜单数据
const menu = ref(Object.assign({}, defaultMenu));
// 选择菜单列表
const selectMenuList = ref([]);
// 菜单表单组件引用
const menuFromRef = ref();
// 菜单表单校验规则
const rules = reactive({
    title: [
        { required: true, message: '请输入菜单名称', trigger: 'blur' },
        { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
    ],
    name: [
        { required: true, message: '请输入前端名称', trigger: 'blur' },
        { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
    ],
    icon: [
        { required: true, message: '请输入前端图标', trigger: 'blur' },
        { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
    ]
});
// 获取选择菜单列表
const getSelectMenuList = async () => {
    const res = await getMenuListByParentIdAPI(0, { pageSize: 100, pageNum: 1 });
    selectMenuList.value = res.data.list;
    const noParentMenu = { ...defaultMenu };
    noParentMenu.id = 0;
    noParentMenu.title = '无上级菜单';
    selectMenuList.value.unshift(noParentMenu);
};
// 组件挂载时加载数据
onMounted(async () => {
    if (props.isEdit) {
        const res = await getMenuByIdAPI(Number(route.query.id));
        menu.value = res.data;
    }
    else {
        menu.value = Object.assign({}, defaultMenu);
    }
    getSelectMenuList();
});
// 处理菜单表单提交
const onSubmit = () => {
    menuFromRef.value.validate(async (valid) => {
        if (valid) {
            await ElMessageBox.confirm('是否提交数据', '提示', {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning'
            });
            if (props.isEdit) {
                await updateMenu(Number(route.query.id), menu.value);
                menuFromRef.value.resetFields();
                ElMessage({
                    message: '修改成功',
                    type: 'success',
                    duration: 1000
                });
                router.back();
            }
            else {
                await menuCreateAPI(menu.value);
                menuFromRef.value.resetFields();
                resetForm();
                ElMessage({
                    message: '提交成功',
                    type: 'success',
                    duration: 1000
                });
                router.back();
            }
        }
        else {
            ElMessage({
                message: '验证失败',
                type: 'error',
                duration: 1000
            });
        }
    });
};
// 处理菜单表单重置
const resetForm = () => {
    menuFromRef.value.resetFields();
    menu.value = Object.assign({}, defaultMenu);
    getSelectMenuList();
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
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_8 = __VLS_asFunctionalComponent1(__VLS_7, new __VLS_7({
    model: (__VLS_ctx.menu),
    rules: (__VLS_ctx.rules),
    ref: "menuFromRef",
    labelWidth: "150px",
}));
const __VLS_9 = __VLS_8({
    model: (__VLS_ctx.menu),
    rules: (__VLS_ctx.rules),
    ref: "menuFromRef",
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_8));
var __VLS_12 = {};
const { default: __VLS_14 } = __VLS_10.slots;
let __VLS_15;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_16 = __VLS_asFunctionalComponent1(__VLS_15, new __VLS_15({
    label: "菜单名称：",
    prop: "title",
}));
const __VLS_17 = __VLS_16({
    label: "菜单名称：",
    prop: "title",
}, ...__VLS_functionalComponentArgsRest(__VLS_16));
const { default: __VLS_20 } = __VLS_18.slots;
let __VLS_21;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_22 = __VLS_asFunctionalComponent1(__VLS_21, new __VLS_21({
    modelValue: (__VLS_ctx.menu.title),
}));
const __VLS_23 = __VLS_22({
    modelValue: (__VLS_ctx.menu.title),
}, ...__VLS_functionalComponentArgsRest(__VLS_22));
// @ts-ignore
[menu, menu, rules,];
var __VLS_18;
let __VLS_26;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_27 = __VLS_asFunctionalComponent1(__VLS_26, new __VLS_26({
    label: "上级菜单：",
}));
const __VLS_28 = __VLS_27({
    label: "上级菜单：",
}, ...__VLS_functionalComponentArgsRest(__VLS_27));
const { default: __VLS_31 } = __VLS_29.slots;
let __VLS_32;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_33 = __VLS_asFunctionalComponent1(__VLS_32, new __VLS_32({
    modelValue: (__VLS_ctx.menu.parentId),
    placeholder: "请选择菜单",
}));
const __VLS_34 = __VLS_33({
    modelValue: (__VLS_ctx.menu.parentId),
    placeholder: "请选择菜单",
}, ...__VLS_functionalComponentArgsRest(__VLS_33));
const { default: __VLS_37 } = __VLS_35.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.selectMenuList))) {
    let __VLS_38;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_39 = __VLS_asFunctionalComponent1(__VLS_38, new __VLS_38({
        key: (item.id),
        label: (item.title),
        value: (item.id),
    }));
    const __VLS_40 = __VLS_39({
        key: (item.id),
        label: (item.title),
        value: (item.id),
    }, ...__VLS_functionalComponentArgsRest(__VLS_39));
    // @ts-ignore
    [menu, selectMenuList,];
}
// @ts-ignore
[];
var __VLS_35;
// @ts-ignore
[];
var __VLS_29;
let __VLS_43;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_44 = __VLS_asFunctionalComponent1(__VLS_43, new __VLS_43({
    label: "前端名称：",
    prop: "name",
}));
const __VLS_45 = __VLS_44({
    label: "前端名称：",
    prop: "name",
}, ...__VLS_functionalComponentArgsRest(__VLS_44));
const { default: __VLS_48 } = __VLS_46.slots;
let __VLS_49;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_50 = __VLS_asFunctionalComponent1(__VLS_49, new __VLS_49({
    modelValue: (__VLS_ctx.menu.name),
}));
const __VLS_51 = __VLS_50({
    modelValue: (__VLS_ctx.menu.name),
}, ...__VLS_functionalComponentArgsRest(__VLS_50));
// @ts-ignore
[menu,];
var __VLS_46;
let __VLS_54;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    label: "前端图标：",
    prop: "icon",
}));
const __VLS_56 = __VLS_55({
    label: "前端图标：",
    prop: "icon",
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
const { default: __VLS_59 } = __VLS_57.slots;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    modelValue: (__VLS_ctx.menu.icon),
    ...{ style: {} },
}));
const __VLS_62 = __VLS_61({
    modelValue: (__VLS_ctx.menu.icon),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
let __VLS_65;
/** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
svgIcon;
// @ts-ignore
const __VLS_66 = __VLS_asFunctionalComponent1(__VLS_65, new __VLS_65({
    ...{ style: {} },
    iconClass: (__VLS_ctx.menu.icon),
}));
const __VLS_67 = __VLS_66({
    ...{ style: {} },
    iconClass: (__VLS_ctx.menu.icon),
}, ...__VLS_functionalComponentArgsRest(__VLS_66));
// @ts-ignore
[menu, menu,];
var __VLS_57;
let __VLS_70;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_71 = __VLS_asFunctionalComponent1(__VLS_70, new __VLS_70({
    label: "是否显示：",
}));
const __VLS_72 = __VLS_71({
    label: "是否显示：",
}, ...__VLS_functionalComponentArgsRest(__VLS_71));
const { default: __VLS_75 } = __VLS_73.slots;
let __VLS_76;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_77 = __VLS_asFunctionalComponent1(__VLS_76, new __VLS_76({
    modelValue: (__VLS_ctx.menu.hidden),
}));
const __VLS_78 = __VLS_77({
    modelValue: (__VLS_ctx.menu.hidden),
}, ...__VLS_functionalComponentArgsRest(__VLS_77));
const { default: __VLS_81 } = __VLS_79.slots;
let __VLS_82;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_83 = __VLS_asFunctionalComponent1(__VLS_82, new __VLS_82({
    label: (0),
}));
const __VLS_84 = __VLS_83({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_83));
const { default: __VLS_87 } = __VLS_85.slots;
// @ts-ignore
[menu,];
var __VLS_85;
let __VLS_88;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_89 = __VLS_asFunctionalComponent1(__VLS_88, new __VLS_88({
    label: (1),
}));
const __VLS_90 = __VLS_89({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_89));
const { default: __VLS_93 } = __VLS_91.slots;
// @ts-ignore
[];
var __VLS_91;
// @ts-ignore
[];
var __VLS_79;
// @ts-ignore
[];
var __VLS_73;
let __VLS_94;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_95 = __VLS_asFunctionalComponent1(__VLS_94, new __VLS_94({
    label: "排序：",
}));
const __VLS_96 = __VLS_95({
    label: "排序：",
}, ...__VLS_functionalComponentArgsRest(__VLS_95));
const { default: __VLS_99 } = __VLS_97.slots;
let __VLS_100;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_101 = __VLS_asFunctionalComponent1(__VLS_100, new __VLS_100({
    modelValue: (__VLS_ctx.menu.sort),
}));
const __VLS_102 = __VLS_101({
    modelValue: (__VLS_ctx.menu.sort),
}, ...__VLS_functionalComponentArgsRest(__VLS_101));
// @ts-ignore
[menu,];
var __VLS_97;
let __VLS_105;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_106 = __VLS_asFunctionalComponent1(__VLS_105, new __VLS_105({}));
const __VLS_107 = __VLS_106({}, ...__VLS_functionalComponentArgsRest(__VLS_106));
const { default: __VLS_110 } = __VLS_108.slots;
let __VLS_111;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_112 = __VLS_asFunctionalComponent1(__VLS_111, new __VLS_111({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_113 = __VLS_112({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_112));
let __VLS_116;
const __VLS_117 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.onSubmit();
            // @ts-ignore
            [onSubmit,];
        } });
const { default: __VLS_118 } = __VLS_114.slots;
// @ts-ignore
[];
var __VLS_114;
var __VLS_115;
if (!props.isEdit) {
    let __VLS_119;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_120 = __VLS_asFunctionalComponent1(__VLS_119, new __VLS_119({
        ...{ 'onClick': {} },
    }));
    const __VLS_121 = __VLS_120({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_120));
    let __VLS_124;
    const __VLS_125 = ({ click: {} },
        { onClick: (...[$event]) => {
                if (!(!props.isEdit))
                    return;
                __VLS_ctx.resetForm();
                // @ts-ignore
                [resetForm,];
            } });
    const { default: __VLS_126 } = __VLS_122.slots;
    // @ts-ignore
    [];
    var __VLS_122;
    var __VLS_123;
}
// @ts-ignore
[];
var __VLS_108;
// @ts-ignore
[];
var __VLS_10;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
var __VLS_13 = __VLS_12;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({
    props: {
        isEdit: {
            type: Boolean,
            default: false
        }
    },
});
export default {};
