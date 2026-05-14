/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { reactive, ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { useUserStore } from '@/stores/user';
import { isvalidUsername } from '@/utils/validate';
// 使用Vue Router和Pinia
const router = useRouter();
const userStore = useUserStore();
// 表单引用
const loginFormRef = ref();
// 登陆表单数据
const loginForm = reactive({
    username: '',
    password: '',
});
// 表单参数校验规则
const loginRules = reactive({
    username: [{ required: true, trigger: 'blur', validator: validateUsername }],
    password: [{ required: true, trigger: 'blur', validator: validatePass }]
});
// 登陆按钮进度条
const loading = ref(false);
// 用户名验证函数
function validateUsername(rule, value, callback) {
    if (!isvalidUsername(value)) {
        callback(new Error('请输入正确的用户名'));
    }
    else {
        callback();
    }
}
;
// 密码验证函数
function validatePass(rule, value, callback) {
    if (value.length < 3) {
        callback(new Error('密码不能小于3位'));
    }
    else {
        callback();
    }
}
;
// 组件挂载完成后调用
onMounted(() => {
    loginForm.username = userStore.userInfo.username;
    loginForm.password = userStore.userInfo.password;
    if (loginForm.username === undefined || loginForm.username == null || loginForm.username === '') {
        loginForm.username = 'admin';
    }
});
// 处理登录按钮事件
const handleLogin = () => {
    loginFormRef.value.validate(async (valid) => {
        if (valid) {
            loading.value = true;
            try {
                await userStore.userLogin({
                    username: loginForm.username.trim(),
                    password: loginForm.password
                });
                loading.value = false;
                router.push({ path: '/' });
            }
            catch (err) {
                loading.value = false;
                console.log(err);
            }
        }
        else {
            console.log('参数验证不合法！');
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
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    ...{ class: "login-form-layout" },
}));
const __VLS_2 = __VLS_1({
    ...{ class: "login-form-layout" },
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
/** @type {__VLS_StyleScopedClasses['login-form-layout']} */ ;
const { default: __VLS_5 } = __VLS_3.slots;
let __VLS_6;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({
    autoComplete: "on",
    model: (__VLS_ctx.loginForm),
    rules: (__VLS_ctx.loginRules),
    ref: "loginFormRef",
    labelPosition: "left",
}));
const __VLS_8 = __VLS_7({
    autoComplete: "on",
    model: (__VLS_ctx.loginForm),
    rules: (__VLS_ctx.loginRules),
    ref: "loginFormRef",
    labelPosition: "left",
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
var __VLS_11 = {};
const { default: __VLS_13 } = __VLS_9.slots;
let __VLS_14;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_15 = __VLS_asFunctionalComponent1(__VLS_14, new __VLS_14({
    prop: "username",
}));
const __VLS_16 = __VLS_15({
    prop: "username",
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
const { default: __VLS_19 } = __VLS_17.slots;
let __VLS_20;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_21 = __VLS_asFunctionalComponent1(__VLS_20, new __VLS_20({
    name: "username",
    type: "text",
    modelValue: (__VLS_ctx.loginForm.username),
    autoComplete: "on",
    placeholder: "请输入用户名",
}));
const __VLS_22 = __VLS_21({
    name: "username",
    type: "text",
    modelValue: (__VLS_ctx.loginForm.username),
    autoComplete: "on",
    placeholder: "请输入用户名",
}, ...__VLS_functionalComponentArgsRest(__VLS_21));
const { default: __VLS_25 } = __VLS_23.slots;
{
    const { prefix: __VLS_26 } = __VLS_23.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
    let __VLS_27;
    /** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
    svgIcon;
    // @ts-ignore
    const __VLS_28 = __VLS_asFunctionalComponent1(__VLS_27, new __VLS_27({
        iconClass: "user",
        ...{ class: "color-main" },
    }));
    const __VLS_29 = __VLS_28({
        iconClass: "user",
        ...{ class: "color-main" },
    }, ...__VLS_functionalComponentArgsRest(__VLS_28));
    /** @type {__VLS_StyleScopedClasses['color-main']} */ ;
    // @ts-ignore
    [loginForm, loginForm, loginRules,];
}
// @ts-ignore
[];
var __VLS_23;
// @ts-ignore
[];
var __VLS_17;
let __VLS_32;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_33 = __VLS_asFunctionalComponent1(__VLS_32, new __VLS_32({
    prop: "password",
}));
const __VLS_34 = __VLS_33({
    prop: "password",
}, ...__VLS_functionalComponentArgsRest(__VLS_33));
const { default: __VLS_37 } = __VLS_35.slots;
let __VLS_38;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_39 = __VLS_asFunctionalComponent1(__VLS_38, new __VLS_38({
    ...{ 'onKeyup': {} },
    name: "password",
    modelValue: (__VLS_ctx.loginForm.password),
    autoComplete: "on",
    showPassword: true,
    placeholder: "请输入密码",
}));
const __VLS_40 = __VLS_39({
    ...{ 'onKeyup': {} },
    name: "password",
    modelValue: (__VLS_ctx.loginForm.password),
    autoComplete: "on",
    showPassword: true,
    placeholder: "请输入密码",
}, ...__VLS_functionalComponentArgsRest(__VLS_39));
let __VLS_43;
const __VLS_44 = ({ keyup: {} },
    { onKeyup: (__VLS_ctx.handleLogin) });
const { default: __VLS_45 } = __VLS_41.slots;
{
    const { prefix: __VLS_46 } = __VLS_41.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
    let __VLS_47;
    /** @ts-ignore @type { | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon'] | typeof __VLS_components.svgIcon | typeof __VLS_components.SvgIcon | typeof __VLS_components['svg-icon']} */
    svgIcon;
    // @ts-ignore
    const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({
        iconClass: "password",
        ...{ class: "color-main" },
    }));
    const __VLS_49 = __VLS_48({
        iconClass: "password",
        ...{ class: "color-main" },
    }, ...__VLS_functionalComponentArgsRest(__VLS_48));
    /** @type {__VLS_StyleScopedClasses['color-main']} */ ;
    // @ts-ignore
    [loginForm, handleLogin,];
}
// @ts-ignore
[];
var __VLS_41;
var __VLS_42;
// @ts-ignore
[];
var __VLS_35;
let __VLS_52;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_53 = __VLS_asFunctionalComponent1(__VLS_52, new __VLS_52({
    ...{ style: {} },
}));
const __VLS_54 = __VLS_53({
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_53));
const { default: __VLS_57 } = __VLS_55.slots;
let __VLS_58;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_59 = __VLS_asFunctionalComponent1(__VLS_58, new __VLS_58({
    ...{ 'onClick': {} },
    ...{ style: {} },
    type: "primary",
    loading: (__VLS_ctx.loading),
}));
const __VLS_60 = __VLS_59({
    ...{ 'onClick': {} },
    ...{ style: {} },
    type: "primary",
    loading: (__VLS_ctx.loading),
}, ...__VLS_functionalComponentArgsRest(__VLS_59));
let __VLS_63;
const __VLS_64 = ({ click: {} },
    { onClick: (__VLS_ctx.handleLogin) });
const { default: __VLS_65 } = __VLS_61.slots;
// @ts-ignore
[handleLogin, loading,];
var __VLS_61;
var __VLS_62;
// @ts-ignore
[];
var __VLS_55;
// @ts-ignore
[];
var __VLS_9;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
var __VLS_12 = __VLS_11;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
