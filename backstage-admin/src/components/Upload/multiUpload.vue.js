/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, watch } from 'vue';
import { Plus } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import { MESSAGE_DURATION_SHORT } from '@/constants';
import { t } from '@/locales';
const props = defineProps({
    modelValue: {
        type: (Array),
        default: []
    },
    maxCount: {
        type: Number,
        default: 5
    }
});
const emit = defineEmits(['update:modelValue']);
const dialogVisible = ref(false);
const dialogImageUrl = ref('');
const uploadUrl = import.meta.env.VITE_BASE_SERVER_URL + import.meta.env.VITE_UPLOAD_URL;
const fileList = ref([]);
watch(() => props.modelValue, (newVal) => {
    if (newVal) {
        fileList.value = newVal.map(item => {
            const fileName = item.substring(item.lastIndexOf("/") + 1);
            return { name: fileName, url: item };
        });
    }
    else {
        fileList.value = [];
    }
}, { immediate: true });
const emitInput = (val) => {
    emit('update:modelValue', val);
};
const handleRemove = (file, fileList) => {
    const remainingFiles = fileList.filter(item => item.uid !== file.uid);
    fileList = remainingFiles;
    const urls = remainingFiles.map(item => item.url || '');
    emitInput(urls);
};
const handlePreview = (file) => {
    dialogVisible.value = true;
    dialogImageUrl.value = file.url;
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
    fileList.value.push({ name: file.name, url: res.data });
    const urls = fileList.value.map(item => item.url || '');
    emitInput(urls);
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
    listType: "picture-card",
    fileList: (__VLS_ctx.fileList),
    limit: (__VLS_ctx.maxCount),
    onRemove: (__VLS_ctx.handleRemove),
    onSuccess: (__VLS_ctx.handleUploadSuccess),
    onPreview: (__VLS_ctx.handlePreview),
}));
const __VLS_2 = __VLS_1({
    action: (__VLS_ctx.uploadUrl),
    listType: "picture-card",
    fileList: (__VLS_ctx.fileList),
    limit: (__VLS_ctx.maxCount),
    onRemove: (__VLS_ctx.handleRemove),
    onSuccess: (__VLS_ctx.handleUploadSuccess),
    onPreview: (__VLS_ctx.handlePreview),
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
const { default: __VLS_5 } = __VLS_3.slots;
let __VLS_6;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({}));
const __VLS_8 = __VLS_7({}, ...__VLS_functionalComponentArgsRest(__VLS_7));
const { default: __VLS_11 } = __VLS_9.slots;
let __VLS_12;
/** @ts-ignore @type { | typeof __VLS_components.Plus} */
Plus;
// @ts-ignore
const __VLS_13 = __VLS_asFunctionalComponent1(__VLS_12, new __VLS_12({}));
const __VLS_14 = __VLS_13({}, ...__VLS_functionalComponentArgsRest(__VLS_13));
// @ts-ignore
[uploadUrl, fileList, maxCount, handleRemove, handleUploadSuccess, handlePreview,];
var __VLS_9;
// @ts-ignore
[];
var __VLS_3;
let __VLS_17;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_18 = __VLS_asFunctionalComponent1(__VLS_17, new __VLS_17({
    modelValue: (__VLS_ctx.dialogVisible),
}));
const __VLS_19 = __VLS_18({
    modelValue: (__VLS_ctx.dialogVisible),
}, ...__VLS_functionalComponentArgsRest(__VLS_18));
const { default: __VLS_22 } = __VLS_20.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.img)({
    src: (__VLS_ctx.dialogImageUrl),
    alt: (__VLS_ctx.dialogImageUrl || 'preview'),
});
// @ts-ignore
[dialogVisible, dialogImageUrl, dialogImageUrl,];
var __VLS_20;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({
    emits: {},
    props: {
        modelValue: {
            type: (Array),
            default: []
        },
        maxCount: {
            type: Number,
            default: 5
        }
    },
});
export default {};
