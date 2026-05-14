/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, watch } from 'vue';
const uploadUrl = import.meta.env.VITE_BASE_SERVER_URL + import.meta.env.VITE_UPLOAD_URL;
const props = withDefaults(defineProps(), {
    disabled: false,
    height: 400,
    toolbar: () => [
        'blocks | undo redo remove| bold italic underline strikethrough removeformat | fontselect fontsizeselect formatselect | alignleft aligncenter alignright alignjustify',
        'fontsize forecolor backcolor bullist numlist |link image table | code fullscreen'
    ],
    plugins: () => [
        'link', 'image', 'table', 'lists', 'code', 'fullscreen'
    ],
    placeholder: ''
});
const emit = defineEmits();
const contentValue = ref(props.modelValue);
// 监听外部值变化
watch(() => props.modelValue, (newValue) => {
    if (newValue !== contentValue.value) {
        contentValue.value = newValue;
    }
});
// 监听内部值变化
watch(contentValue, (newValue) => {
    emit('update:modelValue', newValue);
});
const initOptions = {
    language: 'zh_CN',
    language_url: './tinymce6.8.6/langs/zh_CN.js',
    selector: '#mytextarea',
    height: props.height,
    menubar: false,
    plugins: props.plugins,
    toolbar: props.toolbar,
    placeholder: props.placeholder,
    branding: false, // 隐藏 tinyMCE 品牌标识
    resize: true, // 允许调整大小
    elementpath: false, // 隐藏底部元素路径
    content_style: `
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
      font-size: 14px;
      margin: 10px;
      line-height: 1.6;
    }
  `,
    images_upload_handler: (blobInfo, progress) => new Promise((resolve, reject) => {
        // 发起请求上传图片（目前仅支持minio上传）
        const xhr = new XMLHttpRequest();
        xhr.open('POST', uploadUrl);
        // 上传进度回调
        xhr.upload.onprogress = (e) => {
            progress(e.loaded / e.total * 100);
        };
        // 上传完成回调
        xhr.onload = () => {
            if (xhr.status === 403) {
                reject('HTTP Error: ' + xhr.status);
                return;
            }
            if (xhr.status < 200 || xhr.status >= 300) {
                reject('HTTP Error: ' + xhr.status);
                return;
            }
            const json = JSON.parse(xhr.responseText);
            console.log('images_upload:', json);
            if (!json || json.code !== 200) {
                reject('上传失败！');
                return;
            }
            resolve(json.data.url);
        };
        // 上传失败回调
        xhr.onerror = () => {
            reject('Image upload failed due to a XHR Transport error. Code: ' + xhr.status);
        };
        const formData = new FormData();
        formData.append('file', blobInfo.blob(), blobInfo.filename());
        xhr.send(formData);
    })
};
const onClick = () => {
    emit('click');
};
const onBlur = () => {
    emit('blur');
};
const onFocus = () => {
    emit('focus');
};
const __VLS_defaults = {
    disabled: false,
    height: 400,
    toolbar: () => [
        'blocks | undo redo remove| bold italic underline strikethrough removeformat | fontselect fontsizeselect formatselect | alignleft aligncenter alignright alignjustify',
        'fontsize forecolor backcolor bullist numlist |link image table | code fullscreen'
    ],
    plugins: () => [
        'link', 'image', 'table', 'lists', 'code', 'fullscreen'
    ],
    placeholder: ''
};
const __VLS_ctx = {
    ...{},
    ...{},
    ...{},
    ...{},
    ...{},
};
let __VLS_components;
let __VLS_intrinsics;
let __VLS_directives;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "rich-editor" },
});
/** @type {__VLS_StyleScopedClasses['rich-editor']} */ ;
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.editor | typeof __VLS_components.Editor} */
editor;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    ...{ 'onOnClick': {} },
    ...{ 'onOnBlur': {} },
    ...{ 'onOnFocus': {} },
    modelValue: (__VLS_ctx.contentValue),
    init: (__VLS_ctx.initOptions),
    disabled: (__VLS_ctx.disabled),
    tinymceScriptSrc: "./tinymce6.8.6/tinymce.min.js",
}));
const __VLS_2 = __VLS_1({
    ...{ 'onOnClick': {} },
    ...{ 'onOnBlur': {} },
    ...{ 'onOnFocus': {} },
    modelValue: (__VLS_ctx.contentValue),
    init: (__VLS_ctx.initOptions),
    disabled: (__VLS_ctx.disabled),
    tinymceScriptSrc: "./tinymce6.8.6/tinymce.min.js",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
let __VLS_5;
const __VLS_6 = ({ onClick: {} },
    { onOnClick: (__VLS_ctx.onClick) });
const __VLS_7 = ({ onBlur: {} },
    { onOnBlur: (__VLS_ctx.onBlur) });
const __VLS_8 = ({ onFocus: {} },
    { onOnFocus: (__VLS_ctx.onFocus) });
var __VLS_3;
var __VLS_4;
// @ts-ignore
[contentValue, initOptions, disabled, onClick, onBlur, onFocus,];
const __VLS_export = (await import('vue')).defineComponent({
    __typeEmits: {},
    __defaults: __VLS_defaults,
    __typeProps: {},
});
export default {};
