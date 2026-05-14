/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, watch } from 'vue';
import { ElMessage } from 'element-plus';
import { MESSAGE_DURATION_SHORT } from '@/constants';
import { t } from '@/locales';
const props = defineProps({
    modelValue: {
        type: String,
        default: ''
    }
});
const emit = defineEmits(['update:modelValue']);
const dialogVisible = ref(false);
const uploadUrl = import.meta.env.VITE_BASE_SERVER_URL + import.meta.env.VITE_UPLOAD_URL;
const fileList = ref([]);
watch(() => props.modelValue, (newVal) => {
    if (newVal) {
        const fileName = newVal.substring(newVal.lastIndexOf("/") + 1);
        fileList.value = [{
                name: fileName,
                url: newVal
            }];
    }
    else {
        fileList.value = [];
    }
}, { immediate: true });
const emitInput = (val) => {
    emit('update:modelValue', val);
};
const handleRemove = () => {
    emitInput('');
};
const handlePreview = () => {
    dialogVisible.value = true;
};
const handleUploadSuccess = (res, file) => {
    if (res.code === 500) {
        ElMessage({
            message: t('upload.uploadFailed'),
            type: 'error',
            duration: MESSAGE_DURATION_SHORT,
        });
        return;
    }
    fileList.value.pop();
    fileList.value.push({ name: file.name, url: res.data });
    emitInput(res.data);
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
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elUpload | typeof __VLS_components.ElUpload | typeof __VLS_components['el-upload'] | typeof __VLS_components.elUpload | typeof __VLS_components.ElUpload | typeof __VLS_components['el-upload']} */
elUpload;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    action: (__VLS_ctx.uploadUrl),
    listType: "picture",
    multiple: (false),
    showFileList: (props.modelValue ? true : false),
    fileList: (__VLS_ctx.fileList),
    onRemove: (__VLS_ctx.handleRemove),
    onSuccess: (__VLS_ctx.handleUploadSuccess),
    onPreview: (__VLS_ctx.handlePreview),
}));
const __VLS_2 = __VLS_1({
    action: (__VLS_ctx.uploadUrl),
    listType: "picture",
    multiple: (false),
    showFileList: (props.modelValue ? true : false),
    fileList: (__VLS_ctx.fileList),
    onRemove: (__VLS_ctx.handleRemove),
    onSuccess: (__VLS_ctx.handleUploadSuccess),
    onPreview: (__VLS_ctx.handlePreview),
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
const { default: __VLS_5 } = __VLS_3.slots;
let __VLS_6;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({
    size: "small",
    type: "primary",
}));
const __VLS_8 = __VLS_7({
    size: "small",
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
const { default: __VLS_11 } = __VLS_9.slots;
(__VLS_ctx.t('upload.clickToUpload'));
// @ts-ignore
[uploadUrl, fileList, handleRemove, handleUploadSuccess, handlePreview, t,];
var __VLS_9;
{
    const { tip: __VLS_12 } = __VLS_3.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
        ...{ class: "el-upload__tip" },
    });
    /** @type {__VLS_StyleScopedClasses['el-upload__tip']} */ ;
    (__VLS_ctx.t('upload.tip'));
    // @ts-ignore
    [t,];
}
// @ts-ignore
[];
var __VLS_3;
let __VLS_13;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_14 = __VLS_asFunctionalComponent1(__VLS_13, new __VLS_13({
    modelValue: (__VLS_ctx.dialogVisible),
}));
const __VLS_15 = __VLS_14({
    modelValue: (__VLS_ctx.dialogVisible),
}, ...__VLS_functionalComponentArgsRest(__VLS_14));
const { default: __VLS_18 } = __VLS_16.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.img)({
    width: "100%",
    src: (__VLS_ctx.fileList[0]?.url),
    alt: (__VLS_ctx.fileList[0]?.name || 'preview'),
});
// @ts-ignore
[fileList, fileList, dialogVisible,];
var __VLS_16;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({
    emits: {},
    props: {
        modelValue: {
            type: String,
            default: ''
        }
    },
});
export default {};
