/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { Search, Tickets } from '@element-plus/icons-vue';
import { getFlashListAPI, flashUpdateStatusByIdAPI, flashDeleteByIdAPI, flashCreateAPI, flashUpdateByIdAPI } from '@/apis/flash';
import { formatDate } from '@/utils/datetime';
// 获取路由
const router = useRouter();
// 列表查询参数
const listQuery = ref({
    pageNum: 1,
    pageSize: 10,
    keyword: ''
});
// 列表数据
const list = ref([]);
// 总条数
const total = ref(0);
// 加载张图
const listLoading = ref(false);
// 获取列表数据
const getList = async () => {
    listLoading.value = true;
    try {
        const res = await getFlashListAPI(listQuery.value);
        listLoading.value = false;
        list.value = res.data.list;
        total.value = res.data.total;
    }
    catch (error) {
        listLoading.value = false;
        console.error('获取秒杀活动列表失败:', error);
    }
};
// 组件挂载时获取数据
onMounted(() => {
    getList();
});
// 默认秒杀活动对象
const defaultSmsFlashPromotion = {
    title: '',
    status: 0
};
// 当前操作的秒杀活动对象
const flashPromotion = ref(Object.assign({}, defaultSmsFlashPromotion));
// 编辑框是否可见
const dialogVisible = ref(false);
// 是否为编辑状态
const isEdit = ref(false);
// 重置搜索条件
const handleResetSearch = () => {
    listQuery.value = { pageNum: 0, pageSize: 10 };
};
// 搜索列表
const handleSearchList = () => {
    listQuery.value.pageNum = 1;
    getList();
};
// 每页大小改变
const handleSizeChange = (val) => {
    listQuery.value.pageNum = 1;
    listQuery.value.pageSize = val;
    getList();
};
// 当前页改变
const handleCurrentChange = (val) => {
    listQuery.value.pageNum = val;
    getList();
};
// 添加活动
const handleAdd = () => {
    dialogVisible.value = true;
    isEdit.value = false;
    flashPromotion.value = Object.assign({}, defaultSmsFlashPromotion);
};
// 显示时间段列表
const handleShowSessionList = () => {
    router.push({ path: '/sms/flashSession' });
};
// 状态改变
const handleStatusChange = async (index, row) => {
    try {
        await ElMessageBox.confirm('是否要修改该状态?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await flashUpdateStatusByIdAPI(row.id, { status: row.status });
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
// 删除活动
const handleDelete = async (index, row) => {
    try {
        await ElMessageBox.confirm('是否要删除该活动?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
        });
        await flashDeleteByIdAPI(row.id);
        ElMessage.success('删除成功!');
        getList();
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('删除活动失败:', error);
        }
    }
};
// 更新活动
const handleUpdate = (index, row) => {
    dialogVisible.value = true;
    isEdit.value = true;
    flashPromotion.value = Object.assign({}, row);
};
// 处理对话框确认
const handleDialogConfirm = async () => {
    await ElMessageBox.confirm('是否要确认?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
    });
    try {
        if (isEdit.value) {
            await flashUpdateByIdAPI(flashPromotion.value.id, flashPromotion.value);
            ElMessage.success('修改成功！');
            dialogVisible.value = false;
            getList();
        }
        else {
            await flashCreateAPI(flashPromotion.value);
            ElMessage.success('添加成功！');
            dialogVisible.value = false;
            getList();
        }
    }
    catch (error) {
        if (error !== 'cancel') {
            console.error('处理活动失败:', error);
        }
    }
};
// 选择时间段
const handleSelectSession = (index, row) => {
    router.push({ path: '/sms/selectSession', query: { flashPromotionId: row.id } });
};
// 格式化活动状态
const formatActiveStatus = (row) => {
    const nowDate = new Date();
    const startDate = new Date(row.startDate);
    const endDate = new Date(row.endDate);
    if (nowDate.getTime() >= startDate.getTime() && nowDate.getTime() <= endDate.getTime()) {
        return '活动进行中';
    }
    else if (nowDate.getTime() > endDate.getTime()) {
        return '活动已结束';
    }
    else {
        return '活动未开始';
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
    ...{ class: "filter-container" },
    shadow: "never",
}));
const __VLS_2 = __VLS_1({
    ...{ class: "filter-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
/** @type {__VLS_StyleScopedClasses['filter-container']} */ ;
const { default: __VLS_5 } = __VLS_3.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
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
/** @ts-ignore @type { | typeof __VLS_components.Search} */
Search;
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
    ...{ style: {} },
    type: "primary",
}));
const __VLS_19 = __VLS_18({
    ...{ 'onClick': {} },
    ...{ style: {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_18));
let __VLS_22;
const __VLS_23 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleSearchList();
            // @ts-ignore
            [handleSearchList,];
        } });
const { default: __VLS_24 } = __VLS_20.slots;
// @ts-ignore
[];
var __VLS_20;
var __VLS_21;
let __VLS_25;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_26 = __VLS_asFunctionalComponent1(__VLS_25, new __VLS_25({
    ...{ 'onClick': {} },
    ...{ style: {} },
}));
const __VLS_27 = __VLS_26({
    ...{ 'onClick': {} },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_26));
let __VLS_30;
const __VLS_31 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleResetSearch();
            // @ts-ignore
            [handleResetSearch,];
        } });
const { default: __VLS_32 } = __VLS_28.slots;
// @ts-ignore
[];
var __VLS_28;
var __VLS_29;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_33;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_34 = __VLS_asFunctionalComponent1(__VLS_33, new __VLS_33({
    inline: (true),
    model: (__VLS_ctx.listQuery),
    labelWidth: "140px",
}));
const __VLS_35 = __VLS_34({
    inline: (true),
    model: (__VLS_ctx.listQuery),
    labelWidth: "140px",
}, ...__VLS_functionalComponentArgsRest(__VLS_34));
const { default: __VLS_38 } = __VLS_36.slots;
let __VLS_39;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_40 = __VLS_asFunctionalComponent1(__VLS_39, new __VLS_39({
    label: "活动名称：",
}));
const __VLS_41 = __VLS_40({
    label: "活动名称：",
}, ...__VLS_functionalComponentArgsRest(__VLS_40));
const { default: __VLS_44 } = __VLS_42.slots;
let __VLS_45;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_46 = __VLS_asFunctionalComponent1(__VLS_45, new __VLS_45({
    modelValue: (__VLS_ctx.listQuery.keyword),
    ...{ class: "input-width" },
    placeholder: "活动名称",
    clearable: true,
}));
const __VLS_47 = __VLS_46({
    modelValue: (__VLS_ctx.listQuery.keyword),
    ...{ class: "input-width" },
    placeholder: "活动名称",
    clearable: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_46));
/** @type {__VLS_StyleScopedClasses['input-width']} */ ;
// @ts-ignore
[listQuery, listQuery,];
var __VLS_42;
// @ts-ignore
[];
var __VLS_36;
// @ts-ignore
[];
var __VLS_3;
let __VLS_50;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_51 = __VLS_asFunctionalComponent1(__VLS_50, new __VLS_50({
    ...{ class: "operate-container" },
    shadow: "never",
}));
const __VLS_52 = __VLS_51({
    ...{ class: "operate-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_51));
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
const { default: __VLS_55 } = __VLS_53.slots;
let __VLS_56;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
    ...{ class: "el-icon-middle" },
}));
const __VLS_58 = __VLS_57({
    ...{ class: "el-icon-middle" },
}, ...__VLS_functionalComponentArgsRest(__VLS_57));
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_61 } = __VLS_59.slots;
let __VLS_62;
/** @ts-ignore @type { | typeof __VLS_components.Tickets} */
Tickets;
// @ts-ignore
const __VLS_63 = __VLS_asFunctionalComponent1(__VLS_62, new __VLS_62({}));
const __VLS_64 = __VLS_63({}, ...__VLS_functionalComponentArgsRest(__VLS_63));
// @ts-ignore
[];
var __VLS_59;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
    ...{ style: {} },
}));
const __VLS_69 = __VLS_68({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
let __VLS_72;
const __VLS_73 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleAdd();
            // @ts-ignore
            [handleAdd,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_74 } = __VLS_70.slots;
// @ts-ignore
[];
var __VLS_70;
var __VLS_71;
let __VLS_75;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_76 = __VLS_asFunctionalComponent1(__VLS_75, new __VLS_75({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}));
const __VLS_77 = __VLS_76({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}, ...__VLS_functionalComponentArgsRest(__VLS_76));
let __VLS_80;
const __VLS_81 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleShowSessionList();
            // @ts-ignore
            [handleShowSessionList,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_82 } = __VLS_78.slots;
// @ts-ignore
[];
var __VLS_78;
var __VLS_79;
// @ts-ignore
[];
var __VLS_53;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_83;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_84 = __VLS_asFunctionalComponent1(__VLS_83, new __VLS_83({
    ref: "flashTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_85 = __VLS_84({
    ref: "flashTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_84));
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_88 = {};
const { default: __VLS_90 } = __VLS_86.slots;
let __VLS_91;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_92 = __VLS_asFunctionalComponent1(__VLS_91, new __VLS_91({
    type: "selection",
    width: "60",
    align: "center",
}));
const __VLS_93 = __VLS_92({
    type: "selection",
    width: "60",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_92));
let __VLS_96;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_97 = __VLS_asFunctionalComponent1(__VLS_96, new __VLS_96({
    label: "编号",
    width: "100",
    align: "center",
}));
const __VLS_98 = __VLS_97({
    label: "编号",
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_97));
const { default: __VLS_101 } = __VLS_99.slots;
{
    const { default: __VLS_102 } = __VLS_99.slots;
    const [scope] = __VLS_vSlot(__VLS_102);
    (scope.row.id);
    // @ts-ignore
    [list, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_99;
let __VLS_103;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_104 = __VLS_asFunctionalComponent1(__VLS_103, new __VLS_103({
    label: "活动标题",
    align: "center",
}));
const __VLS_105 = __VLS_104({
    label: "活动标题",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_104));
const { default: __VLS_108 } = __VLS_106.slots;
{
    const { default: __VLS_109 } = __VLS_106.slots;
    const [scope] = __VLS_vSlot(__VLS_109);
    (scope.row.title);
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_106;
let __VLS_110;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_111 = __VLS_asFunctionalComponent1(__VLS_110, new __VLS_110({
    label: "活动状态",
    width: "140",
    align: "center",
}));
const __VLS_112 = __VLS_111({
    label: "活动状态",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_111));
const { default: __VLS_115 } = __VLS_113.slots;
{
    const { default: __VLS_116 } = __VLS_113.slots;
    const [scope] = __VLS_vSlot(__VLS_116);
    (__VLS_ctx.formatActiveStatus(scope.row));
    // @ts-ignore
    [formatActiveStatus,];
}
// @ts-ignore
[];
var __VLS_113;
let __VLS_117;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_118 = __VLS_asFunctionalComponent1(__VLS_117, new __VLS_117({
    label: "开始时间",
    width: "140",
    align: "center",
}));
const __VLS_119 = __VLS_118({
    label: "开始时间",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_118));
const { default: __VLS_122 } = __VLS_120.slots;
{
    const { default: __VLS_123 } = __VLS_120.slots;
    const [scope] = __VLS_vSlot(__VLS_123);
    (__VLS_ctx.formatDate(scope.row.startDate));
    // @ts-ignore
    [formatDate,];
}
// @ts-ignore
[];
var __VLS_120;
let __VLS_124;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_125 = __VLS_asFunctionalComponent1(__VLS_124, new __VLS_124({
    label: "结束时间",
    width: "140",
    align: "center",
}));
const __VLS_126 = __VLS_125({
    label: "结束时间",
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_125));
const { default: __VLS_129 } = __VLS_127.slots;
{
    const { default: __VLS_130 } = __VLS_127.slots;
    const [scope] = __VLS_vSlot(__VLS_130);
    (__VLS_ctx.formatDate(scope.row.endDate));
    // @ts-ignore
    [formatDate,];
}
// @ts-ignore
[];
var __VLS_127;
let __VLS_131;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_132 = __VLS_asFunctionalComponent1(__VLS_131, new __VLS_131({
    label: "上线/下线",
    width: "200",
    align: "center",
}));
const __VLS_133 = __VLS_132({
    label: "上线/下线",
    width: "200",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_132));
const { default: __VLS_136 } = __VLS_134.slots;
{
    const { default: __VLS_137 } = __VLS_134.slots;
    const [scope] = __VLS_vSlot(__VLS_137);
    let __VLS_138;
    /** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
    elSwitch;
    // @ts-ignore
    const __VLS_139 = __VLS_asFunctionalComponent1(__VLS_138, new __VLS_138({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.status),
    }));
    const __VLS_140 = __VLS_139({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.status),
    }, ...__VLS_functionalComponentArgsRest(__VLS_139));
    let __VLS_143;
    const __VLS_144 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handleStatusChange(scope.$index, scope.row);
                // @ts-ignore
                [handleStatusChange,];
            } });
    var __VLS_141;
    var __VLS_142;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_134;
let __VLS_145;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_146 = __VLS_asFunctionalComponent1(__VLS_145, new __VLS_145({
    label: "操作",
    width: "180",
    align: "center",
}));
const __VLS_147 = __VLS_146({
    label: "操作",
    width: "180",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_146));
const { default: __VLS_150 } = __VLS_148.slots;
{
    const { default: __VLS_151 } = __VLS_148.slots;
    const [scope] = __VLS_vSlot(__VLS_151);
    let __VLS_152;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_153 = __VLS_asFunctionalComponent1(__VLS_152, new __VLS_152({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_154 = __VLS_153({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_153));
    let __VLS_157;
    const __VLS_158 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleSelectSession(scope.$index, scope.row);
                // @ts-ignore
                [handleSelectSession,];
            } });
    const { default: __VLS_159 } = __VLS_155.slots;
    // @ts-ignore
    [];
    var __VLS_155;
    var __VLS_156;
    let __VLS_160;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_161 = __VLS_asFunctionalComponent1(__VLS_160, new __VLS_160({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_162 = __VLS_161({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_161));
    let __VLS_165;
    const __VLS_166 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdate(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdate,];
            } });
    const { default: __VLS_167 } = __VLS_163.slots;
    // @ts-ignore
    [];
    var __VLS_163;
    var __VLS_164;
    let __VLS_168;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_169 = __VLS_asFunctionalComponent1(__VLS_168, new __VLS_168({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }));
    const __VLS_170 = __VLS_169({
        ...{ 'onClick': {} },
        size: "small",
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_169));
    let __VLS_173;
    const __VLS_174 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_175 } = __VLS_171.slots;
    // @ts-ignore
    [];
    var __VLS_171;
    var __VLS_172;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_148;
// @ts-ignore
[];
var __VLS_86;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_176;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_177 = __VLS_asFunctionalComponent1(__VLS_176, new __VLS_176({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}));
const __VLS_178 = __VLS_177({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    currentPage: (__VLS_ctx.listQuery.pageNum),
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: ([5, 10, 15]),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_177));
let __VLS_181;
const __VLS_182 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_183 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_179;
var __VLS_180;
let __VLS_184;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_185 = __VLS_asFunctionalComponent1(__VLS_184, new __VLS_184({
    title: "添加活动",
    modelValue: (__VLS_ctx.dialogVisible),
    width: "40%",
}));
const __VLS_186 = __VLS_185({
    title: "添加活动",
    modelValue: (__VLS_ctx.dialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_185));
const { default: __VLS_189 } = __VLS_187.slots;
let __VLS_190;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_191 = __VLS_asFunctionalComponent1(__VLS_190, new __VLS_190({
    model: (__VLS_ctx.flashPromotion),
    ref: "SmsFlashPromotionForm",
    labelWidth: "150px",
}));
const __VLS_192 = __VLS_191({
    model: (__VLS_ctx.flashPromotion),
    ref: "SmsFlashPromotionForm",
    labelWidth: "150px",
}, ...__VLS_functionalComponentArgsRest(__VLS_191));
var __VLS_195 = {};
const { default: __VLS_197 } = __VLS_193.slots;
let __VLS_198;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_199 = __VLS_asFunctionalComponent1(__VLS_198, new __VLS_198({
    label: "活动标题：",
}));
const __VLS_200 = __VLS_199({
    label: "活动标题：",
}, ...__VLS_functionalComponentArgsRest(__VLS_199));
const { default: __VLS_203 } = __VLS_201.slots;
let __VLS_204;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_205 = __VLS_asFunctionalComponent1(__VLS_204, new __VLS_204({
    modelValue: (__VLS_ctx.flashPromotion.title),
    ...{ style: {} },
}));
const __VLS_206 = __VLS_205({
    modelValue: (__VLS_ctx.flashPromotion.title),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_205));
// @ts-ignore
[listQuery, listQuery, total, handleSizeChange, handleCurrentChange, dialogVisible, flashPromotion, flashPromotion,];
var __VLS_201;
let __VLS_209;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_210 = __VLS_asFunctionalComponent1(__VLS_209, new __VLS_209({
    label: "开始时间：",
}));
const __VLS_211 = __VLS_210({
    label: "开始时间：",
}, ...__VLS_functionalComponentArgsRest(__VLS_210));
const { default: __VLS_214 } = __VLS_212.slots;
let __VLS_215;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_216 = __VLS_asFunctionalComponent1(__VLS_215, new __VLS_215({
    modelValue: (__VLS_ctx.flashPromotion.startDate),
    type: "date",
    placeholder: "请选择时间",
}));
const __VLS_217 = __VLS_216({
    modelValue: (__VLS_ctx.flashPromotion.startDate),
    type: "date",
    placeholder: "请选择时间",
}, ...__VLS_functionalComponentArgsRest(__VLS_216));
// @ts-ignore
[flashPromotion,];
var __VLS_212;
let __VLS_220;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_221 = __VLS_asFunctionalComponent1(__VLS_220, new __VLS_220({
    label: "结束时间：",
}));
const __VLS_222 = __VLS_221({
    label: "结束时间：",
}, ...__VLS_functionalComponentArgsRest(__VLS_221));
const { default: __VLS_225 } = __VLS_223.slots;
let __VLS_226;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_227 = __VLS_asFunctionalComponent1(__VLS_226, new __VLS_226({
    modelValue: (__VLS_ctx.flashPromotion.endDate),
    type: "date",
    placeholder: "请选择时间",
}));
const __VLS_228 = __VLS_227({
    modelValue: (__VLS_ctx.flashPromotion.endDate),
    type: "date",
    placeholder: "请选择时间",
}, ...__VLS_functionalComponentArgsRest(__VLS_227));
// @ts-ignore
[flashPromotion,];
var __VLS_223;
let __VLS_231;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_232 = __VLS_asFunctionalComponent1(__VLS_231, new __VLS_231({
    label: "上线/下线",
}));
const __VLS_233 = __VLS_232({
    label: "上线/下线",
}, ...__VLS_functionalComponentArgsRest(__VLS_232));
const { default: __VLS_236 } = __VLS_234.slots;
let __VLS_237;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_238 = __VLS_asFunctionalComponent1(__VLS_237, new __VLS_237({
    modelValue: (__VLS_ctx.flashPromotion.status),
}));
const __VLS_239 = __VLS_238({
    modelValue: (__VLS_ctx.flashPromotion.status),
}, ...__VLS_functionalComponentArgsRest(__VLS_238));
const { default: __VLS_242 } = __VLS_240.slots;
let __VLS_243;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_244 = __VLS_asFunctionalComponent1(__VLS_243, new __VLS_243({
    label: (1),
}));
const __VLS_245 = __VLS_244({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_244));
const { default: __VLS_248 } = __VLS_246.slots;
// @ts-ignore
[flashPromotion,];
var __VLS_246;
let __VLS_249;
/** @ts-ignore @type { | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio'] | typeof __VLS_components.elRadio | typeof __VLS_components.ElRadio | typeof __VLS_components['el-radio']} */
elRadio;
// @ts-ignore
const __VLS_250 = __VLS_asFunctionalComponent1(__VLS_249, new __VLS_249({
    label: (0),
}));
const __VLS_251 = __VLS_250({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_250));
const { default: __VLS_254 } = __VLS_252.slots;
// @ts-ignore
[];
var __VLS_252;
// @ts-ignore
[];
var __VLS_240;
// @ts-ignore
[];
var __VLS_234;
// @ts-ignore
[];
var __VLS_193;
{
    const { footer: __VLS_255 } = __VLS_187.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_256;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_257 = __VLS_asFunctionalComponent1(__VLS_256, new __VLS_256({
        ...{ 'onClick': {} },
    }));
    const __VLS_258 = __VLS_257({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_257));
    let __VLS_261;
    const __VLS_262 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.dialogVisible = false;
                // @ts-ignore
                [dialogVisible,];
            } });
    const { default: __VLS_263 } = __VLS_259.slots;
    // @ts-ignore
    [];
    var __VLS_259;
    var __VLS_260;
    let __VLS_264;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_265 = __VLS_asFunctionalComponent1(__VLS_264, new __VLS_264({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_266 = __VLS_265({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_265));
    let __VLS_269;
    const __VLS_270 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDialogConfirm();
                // @ts-ignore
                [handleDialogConfirm,];
            } });
    const { default: __VLS_271 } = __VLS_267.slots;
    // @ts-ignore
    [];
    var __VLS_267;
    var __VLS_268;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_187;
// @ts-ignore
var __VLS_89 = __VLS_88, __VLS_196 = __VLS_195;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
