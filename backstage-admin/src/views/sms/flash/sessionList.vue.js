/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { Tickets } from '@element-plus/icons-vue';
import { getFlashSessionListAPI, flashSessionUpdateStatusByIdAPI, flashSessionDeleteByIdAPI, flashSessionCreateAPI, flashSessionUpdateByIdAPI } from '@/apis/flashSession';
import { formatTime } from '@/utils/datetime';
// 秒杀时间段列表数据
const list = ref([]);
// 加载状态
const listLoading = ref(false);
// 获取列表数据
const getList = async () => {
    listLoading.value = true;
    try {
        const res = await getFlashSessionListAPI();
        listLoading.value = false;
        list.value = res.data;
    }
    catch (error) {
        listLoading.value = false;
        console.error('获取秒杀时间段列表失败:', error);
    }
};
// 组件挂载时获取数据
onMounted(() => {
    getList();
});
// 默认秒杀场次对象
const defaultFlashSession = {
    name: '',
    status: 0
};
// 当前操作的秒杀场次
const flashSession = ref(Object.assign({}, defaultFlashSession));
// 编辑框显示状态
const dialogVisible = ref(false);
// 是否为编辑模式
const isEdit = ref(false);
// 添加时间段
const handleAdd = () => {
    dialogVisible.value = true;
    isEdit.value = false;
    flashSession.value = Object.assign({}, defaultFlashSession);
};
// 状态改变
const handleStatusChange = async (index, row) => {
    try {
        await ElMessageBox.confirm('是否要修改该状态?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await flashSessionUpdateStatusByIdAPI(row.id, { status: row.status });
        ElMessage.success('修改成功!');
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('更新状态失败:', error);
        }
        ElMessage.info('取消修改');
        getList();
    }
};
// 更新时间段
const handleUpdate = (index, row) => {
    dialogVisible.value = true;
    isEdit.value = true;
    flashSession.value = Object.assign({}, row);
    if (row.startTime) {
        flashSession.value.startTime = row.startTime;
    }
    if (row.endTime) {
        flashSession.value.endTime = row.endTime;
    }
};
// 删除时间段
const handleDelete = async (index, row) => {
    try {
        await ElMessageBox.confirm('是否要删除该时间段?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await flashSessionDeleteByIdAPI(row.id);
        ElMessage.success('删除成功!');
        getList();
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('删除时间段失败:', error);
        }
    }
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
            await flashSessionUpdateByIdAPI(flashSession.value.id, flashSession.value);
            ElMessage.success('修改成功！');
            dialogVisible.value = false;
            getList();
        }
        else {
            await flashSessionCreateAPI(flashSession.value);
            ElMessage.success('添加成功！');
            dialogVisible.value = false;
            getList();
        }
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('处理时间段失败:', error);
        }
    }
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
    shadow: "never",
    ...{ class: "operate-container" },
}));
const __VLS_2 = __VLS_1({
    shadow: "never",
    ...{ class: "operate-container" },
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
const { default: __VLS_5 } = __VLS_3.slots;
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
/** @ts-ignore @type { | typeof __VLS_components.Tickets} */
Tickets;
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
    ...{ class: "btn-add" },
}));
const __VLS_19 = __VLS_18({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}, ...__VLS_functionalComponentArgsRest(__VLS_18));
let __VLS_22;
const __VLS_23 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleAdd();
            // @ts-ignore
            [handleAdd,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_24 } = __VLS_20.slots;
// @ts-ignore
[];
var __VLS_20;
var __VLS_21;
// @ts-ignore
[];
var __VLS_3;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_25;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_26 = __VLS_asFunctionalComponent1(__VLS_25, new __VLS_25({
    ref: "flashSessionTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_27 = __VLS_26({
    ref: "flashSessionTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_26));
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_30 = {};
const { default: __VLS_32 } = __VLS_28.slots;
let __VLS_33;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_34 = __VLS_asFunctionalComponent1(__VLS_33, new __VLS_33({
    label: "编号",
    width: "100",
    align: "center",
}));
const __VLS_35 = __VLS_34({
    label: "编号",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_34));
const { default: __VLS_38 } = __VLS_36.slots;
{
    const { default: __VLS_39 } = __VLS_36.slots;
    const [scope] = __VLS_vSlot(__VLS_39);
    (scope.row.id);
    // @ts-ignore
    [list, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_36;
let __VLS_40;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_41 = __VLS_asFunctionalComponent1(__VLS_40, new __VLS_40({
    label: "秒杀时间段名称",
    align: "center",
}));
const __VLS_42 = __VLS_41({
    label: "秒杀时间段名称",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_41));
const { default: __VLS_45 } = __VLS_43.slots;
{
    const { default: __VLS_46 } = __VLS_43.slots;
    const [scope] = __VLS_vSlot(__VLS_46);
    (scope.row.name);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_43;
let __VLS_47;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({
    label: "每日开始时间",
    align: "center",
}));
const __VLS_49 = __VLS_48({
    label: "每日开始时间",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_48));
const { default: __VLS_52 } = __VLS_50.slots;
{
    const { default: __VLS_53 } = __VLS_50.slots;
    const [scope] = __VLS_vSlot(__VLS_53);
    (__VLS_ctx.formatTime(scope.row.startTime));
    // @ts-ignore
    [formatTime,];
}
// @ts-ignore
[];
var __VLS_50;
let __VLS_54;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    label: "每日结束时间",
    align: "center",
}));
const __VLS_56 = __VLS_55({
    label: "每日结束时间",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
const { default: __VLS_59 } = __VLS_57.slots;
{
    const { default: __VLS_60 } = __VLS_57.slots;
    const [scope] = __VLS_vSlot(__VLS_60);
    (__VLS_ctx.formatTime(scope.row.endTime));
    // @ts-ignore
    [formatTime,];
}
// @ts-ignore
[];
var __VLS_57;
let __VLS_61;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_62 = __VLS_asFunctionalComponent1(__VLS_61, new __VLS_61({
    label: "启用",
    align: "center",
}));
const __VLS_63 = __VLS_62({
    label: "启用",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_62));
const { default: __VLS_66 } = __VLS_64.slots;
{
    const { default: __VLS_67 } = __VLS_64.slots;
    const [scope] = __VLS_vSlot(__VLS_67);
    let __VLS_68;
    /** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
    elSwitch;
    // @ts-ignore
    const __VLS_69 = __VLS_asFunctionalComponent1(__VLS_68, new __VLS_68({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.status),
    }));
    const __VLS_70 = __VLS_69({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.status),
    }, ...__VLS_functionalComponentArgsRest(__VLS_69));
    let __VLS_73;
    const __VLS_74 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handleStatusChange(scope.$index, scope.row);
                // @ts-ignore
                [handleStatusChange,];
            } });
    var __VLS_71;
    var __VLS_72;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_64;
let __VLS_75;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_76 = __VLS_asFunctionalComponent1(__VLS_75, new __VLS_75({
    label: "操作",
    width: "180",
    align: "center",
}));
const __VLS_77 = __VLS_76({
    label: "操作",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_76));
const { default: __VLS_80 } = __VLS_78.slots;
{
    const { default: __VLS_81 } = __VLS_78.slots;
    const [scope] = __VLS_vSlot(__VLS_81);
    let __VLS_82;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_83 = __VLS_asFunctionalComponent1(__VLS_82, new __VLS_82({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_84 = __VLS_83({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_83));
    let __VLS_87;
    const __VLS_88 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdate(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdate,];
            } });
    const { default: __VLS_89 } = __VLS_85.slots;
    // @ts-ignore
    [];
    var __VLS_85;
    var __VLS_86;
    let __VLS_90;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_91 = __VLS_asFunctionalComponent1(__VLS_90, new __VLS_90({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_92 = __VLS_91({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_91));
    let __VLS_95;
    const __VLS_96 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_97 } = __VLS_93.slots;
    // @ts-ignore
    [];
    var __VLS_93;
    var __VLS_94;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_78;
// @ts-ignore
[];
var __VLS_28;
let __VLS_98;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_99 = __VLS_asFunctionalComponent1(__VLS_98, new __VLS_98({
    title: "添加时间段",
    modelValue: (__VLS_ctx.dialogVisible),
    width: "40%",
}));
const __VLS_100 = __VLS_99({
    title: "添加时间段",
    modelValue: (__VLS_ctx.dialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_99));
const { default: __VLS_103 } = __VLS_101.slots;
let __VLS_104;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_105 = __VLS_asFunctionalComponent1(__VLS_104, new __VLS_104({
    model: (__VLS_ctx.flashSession),
    ref: "flashSessionForm",
    labelWidth: "150px",
}));
const __VLS_106 = __VLS_105({
    model: (__VLS_ctx.flashSession),
    ref: "flashSessionForm",
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_105));
var __VLS_109 = {};
const { default: __VLS_111 } = __VLS_107.slots;
let __VLS_112;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_113 = __VLS_asFunctionalComponent1(__VLS_112, new __VLS_112({
    label: "秒杀时间段名称：",
}));
const __VLS_114 = __VLS_113({
    label: "秒杀时间段名称：",
}, ...__VLS_functionalComponentArgsRest(__VLS_113));
const { default: __VLS_117 } = __VLS_115.slots;
let __VLS_118;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_119 = __VLS_asFunctionalComponent1(__VLS_118, new __VLS_118({
    modelValue: (__VLS_ctx.flashSession.name),
    ...{ style: {} },
}));
const __VLS_120 = __VLS_119({
    modelValue: (__VLS_ctx.flashSession.name),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_119));
// @ts-ignore
[dialogVisible, flashSession, flashSession,];
var __VLS_115;
let __VLS_123;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_124 = __VLS_asFunctionalComponent1(__VLS_123, new __VLS_123({
    label: "每日开始时间：",
}));
const __VLS_125 = __VLS_124({
    label: "每日开始时间：",
}, ...__VLS_functionalComponentArgsRest(__VLS_124));
const { default: __VLS_128 } = __VLS_126.slots;
let __VLS_129;
/** @ts-ignore @type { | typeof __VLS_components.elTimePicker | typeof __VLS_components.ElTimePicker | typeof __VLS_components['el-time-picker'] | typeof __VLS_components.elTimePicker | typeof __VLS_components.ElTimePicker | typeof __VLS_components['el-time-picker']} */
elTimePicker;
// @ts-ignore
const __VLS_130 = __VLS_asFunctionalComponent1(__VLS_129, new __VLS_129({
    modelValue: (__VLS_ctx.flashSession.startTime),
    placeholder: "请选择时间",
}));
const __VLS_131 = __VLS_130({
    modelValue: (__VLS_ctx.flashSession.startTime),
    placeholder: "请选择时间",
}, ...__VLS_functionalComponentArgsRest(__VLS_130));
// @ts-ignore
[flashSession,];
var __VLS_126;
let __VLS_134;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_135 = __VLS_asFunctionalComponent1(__VLS_134, new __VLS_134({
    label: "每日结束时间：",
}));
const __VLS_136 = __VLS_135({
    label: "每日结束时间：",
}, ...__VLS_functionalComponentArgsRest(__VLS_135));
const { default: __VLS_139 } = __VLS_137.slots;
let __VLS_140;
/** @ts-ignore @type { | typeof __VLS_components.elTimePicker | typeof __VLS_components.ElTimePicker | typeof __VLS_components['el-time-picker'] | typeof __VLS_components.elTimePicker | typeof __VLS_components.ElTimePicker | typeof __VLS_components['el-time-picker']} */
elTimePicker;
// @ts-ignore
const __VLS_141 = __VLS_asFunctionalComponent1(__VLS_140, new __VLS_140({
    modelValue: (__VLS_ctx.flashSession.endTime),
    placeholder: "请选择时间",
}));
const __VLS_142 = __VLS_141({
    modelValue: (__VLS_ctx.flashSession.endTime),
    placeholder: "请选择时间",
}, ...__VLS_functionalComponentArgsRest(__VLS_141));
// @ts-ignore
[flashSession,];
var __VLS_137;
let __VLS_145;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_146 = __VLS_asFunctionalComponent1(__VLS_145, new __VLS_145({
    label: "是否启用",
}));
const __VLS_147 = __VLS_146({
    label: "是否启用",
}, ...__VLS_functionalComponentArgsRest(__VLS_146));
const { default: __VLS_150 } = __VLS_148.slots;
let __VLS_151;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_152 = __VLS_asFunctionalComponent1(__VLS_151, new __VLS_151({
    modelValue: (__VLS_ctx.flashSession.status),
}));
const __VLS_153 = __VLS_152({
    modelValue: (__VLS_ctx.flashSession.status),
}, ...__VLS_functionalComponentArgsRest(__VLS_152));
const { default: __VLS_156 } = __VLS_154.slots;
let __VLS_157;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_158 = __VLS_asFunctionalComponent1(__VLS_157, new __VLS_157({
    label: (1),
}));
const __VLS_159 = __VLS_158({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_158));
const { default: __VLS_162 } = __VLS_160.slots;
// @ts-ignore
[flashSession,];
var __VLS_160;
let __VLS_163;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_164 = __VLS_asFunctionalComponent1(__VLS_163, new __VLS_163({
    label: (0),
}));
const __VLS_165 = __VLS_164({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_164));
const { default: __VLS_168 } = __VLS_166.slots;
// @ts-ignore
[];
var __VLS_166;
// @ts-ignore
[];
var __VLS_154;
// @ts-ignore
[];
var __VLS_148;
// @ts-ignore
[];
var __VLS_107;
{
    const { footer: __VLS_169 } = __VLS_101.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_170;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_171 = __VLS_asFunctionalComponent1(__VLS_170, new __VLS_170({
        ...{ 'onClick': {} },
    }));
    const __VLS_172 = __VLS_171({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_171));
    let __VLS_175;
    const __VLS_176 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.dialogVisible = false;
                // @ts-ignore
                [dialogVisible,];
            } });
    const { default: __VLS_177 } = __VLS_173.slots;
    // @ts-ignore
    [];
    var __VLS_173;
    var __VLS_174;
    let __VLS_178;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_179 = __VLS_asFunctionalComponent1(__VLS_178, new __VLS_178({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_180 = __VLS_179({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_179));
    let __VLS_183;
    const __VLS_184 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDialogConfirm();
                // @ts-ignore
                [handleDialogConfirm,];
            } });
    const { default: __VLS_185 } = __VLS_181.slots;
    // @ts-ignore
    [];
    var __VLS_181;
    var __VLS_182;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_101;
// @ts-ignore
var __VLS_31 = __VLS_30, __VLS_110 = __VLS_109;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
