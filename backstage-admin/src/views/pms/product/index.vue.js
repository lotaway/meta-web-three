/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted, reactive, watch } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { getProductListAPI, productUpdateDeleteStatusAPI, productUpdateNewStatusAPI, productUpdateRecommendStatusAPI, productUpdatePublishStatusAPI } from '@/apis/product';
import { getSkuListByPidAPI, skuUpdateByPidAPI } from '@/apis/skuStock';
import { getProductAttributeListAPI } from '@/apis/productAttr';
import { getBrandListAPI } from '@/apis/brand';
import { getProductCategoryListWithChildrenAPI } from '@/apis/productCate';
import { Search, Tickets, Edit } from '@element-plus/icons-vue';
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants';
import { t } from '@/locales';
const router = useRouter();
const listQuery = ref({
    pageNum: 1,
    pageSize: DEFAULT_PAGE_SIZE
});
const list = ref([]);
const total = ref(0);
const listLoading = ref(true);
const getList = async () => {
    listLoading.value = true;
    try {
        const response = await getProductListAPI(listQuery.value);
        listLoading.value = false;
        list.value = response.data.list;
        total.value = response.data.total;
    }
    catch {
        listLoading.value = false;
    }
};
const brandOptions = ref([]);
const getBrandList = async () => {
    const res = await getBrandListAPI({ pageNum: 1, pageSize: 100 });
    brandOptions.value = res.data.list.map(item => ({ label: item.name, value: item.id.toString() }));
};
const productCateOptions = ref([]);
const selectProductCateValue = ref([]);
const getProductCateList = async () => {
    const res = await getProductCategoryListWithChildrenAPI();
    const list = res.data;
    productCateOptions.value = list.map(item => ({
        label: item.name,
        value: item.id,
        children: item.children?.map(it => ({ label: it.name, value: it.id }))
    }));
};
const publishStatusOptions = ref([
    { value: 1, label: t('product.onSale') },
    { value: 0, label: t('product.offSale') }
]);
const verifyStatusOptions = ref([
    { value: 1, label: t('product.verifyPassed') },
    { value: 0, label: t('product.notVerified') }
]);
watch(selectProductCateValue, (newValue) => {
    if (newValue != null && newValue.length == 2) {
        listQuery.value.productCategoryId = newValue[1];
    }
    else {
        listQuery.value.productCategoryId = undefined;
    }
}, { immediate: true });
onMounted(() => {
    getList();
    getBrandList();
    getProductCateList();
});
const operates = ref([
    { label: t('product.productOnSale'), value: 'publishOn' },
    { label: t('product.productOffSale'), value: 'publishOff' },
    { label: t('product.setRecommend'), value: 'recommendOn' },
    { label: t('product.cancelRecommend'), value: 'recommendOff' },
    { label: t('product.setNew'), value: 'newOn' },
    { label: t('product.cancelNew'), value: 'newOff' },
    { label: t('product.moveToRecycleBin'), value: 'recycle' },
    { label: t('product.transferCategory'), value: 'transferCategory' }
]);
const operateType = ref();
const multipleSelection = ref([]);
const editSkuInfo = reactive({
    dialogVisible: false,
    productId: 0,
    productSn: '',
    productAttributeCategoryId: 0,
    stockList: [],
    productAttr: [],
    keyword: undefined
});
const getProductSkuSp = (row, index) => {
    const spData = JSON.parse(row.spData);
    if (spData && index < spData.length) {
        return spData[index].value;
    }
    else {
        return '';
    }
};
const handleShowSkuEditDialog = async (index, row) => {
    editSkuInfo.dialogVisible = true;
    editSkuInfo.productId = row.id;
    editSkuInfo.productSn = row.productSn;
    editSkuInfo.productAttributeCategoryId = row.productAttributeCategoryId;
    editSkuInfo.keyword = undefined;
    const resp = await getSkuListByPidAPI(row.id, { keyword: editSkuInfo.keyword });
    editSkuInfo.stockList = resp.data;
    if (row.productAttributeCategoryId) {
        const res2 = await getProductAttributeListAPI(row.productAttributeCategoryId, { pageNum: 1, pageSize: 10, type: 0 });
        editSkuInfo.productAttr = res2.data.list;
    }
};
const handleSearchEditSku = async () => {
    const response = await getSkuListByPidAPI(editSkuInfo.productId, { keyword: editSkuInfo.keyword });
    editSkuInfo.stockList = response.data;
};
const handleEditSkuConfirm = async () => {
    if (!editSkuInfo.stockList || editSkuInfo.stockList.length <= 0) {
        ElMessage({
            message: t('message.noSkuInfo'),
            type: 'warning',
            duration: MESSAGE_DURATION_SHORT
        });
        return;
    }
    await ElMessageBox.confirm(t('message.confirmModify'), t('common.warning'), {
        confirmButtonText: t('common.confirm'),
        cancelButtonText: t('common.cancel'),
        type: 'warning'
    });
    await skuUpdateByPidAPI(editSkuInfo.productId, editSkuInfo.stockList);
    ElMessage({
        message: t('message.modifySuccess'),
        type: 'success',
        duration: MESSAGE_DURATION_SHORT
    });
    editSkuInfo.dialogVisible = false;
};
const handleSearchList = () => {
    listQuery.value.pageNum = 1;
    getList();
};
const handleAddProduct = () => {
    router.push({ path: '/pms/addProduct' });
};
const handleBatchOperate = async () => {
    if (!operateType.value) {
        ElMessage({
            message: t('message.selectOperationType'),
            type: 'warning',
            duration: MESSAGE_DURATION_SHORT
        });
        return;
    }
    if (!multipleSelection.value || multipleSelection.value.length < 1) {
        ElMessage({
            message: t('message.selectOperateProduct'),
            type: 'warning',
            duration: MESSAGE_DURATION_SHORT
        });
        return;
    }
    await ElMessageBox.confirm(t('message.confirmBatchOperation'), t('common.warning'), {
        confirmButtonText: t('common.confirm'),
        cancelButtonText: t('common.cancel'),
        type: 'warning'
    });
    const ids = multipleSelection.value.map(item => item.id);
    switch (operateType.value) {
        case operates.value[0].value:
            updatePublishStatus(1, ids);
            break;
        case operates.value[1].value:
            updatePublishStatus(0, ids);
            break;
        case operates.value[2].value:
            updateRecommendStatus(1, ids);
            break;
        case operates.value[3].value:
            updateRecommendStatus(0, ids);
            break;
        case operates.value[4].value:
            updateNewStatus(1, ids);
            break;
        case operates.value[5].value:
            updateNewStatus(0, ids);
            break;
        case operates.value[6].value:
            updateDeleteStatus(1, ids);
            break;
        case operates.value[7].value:
            break;
        default:
            break;
    }
    getList();
};
const handleSizeChange = (val) => {
    listQuery.value.pageNum = 1;
    listQuery.value.pageSize = val;
    getList();
};
const handleCurrentChange = (val) => {
    listQuery.value.pageNum = val;
    getList();
};
const handleSelectionChange = (val) => {
    multipleSelection.value = val;
};
const handlePublishStatusChange = async (index, row) => {
    await updatePublishStatus(row.publishStatus, [row.id]);
};
const handleNewStatusChange = async (index, row) => {
    await updateNewStatus(row.newStatus, [row.id]);
};
const handleRecommendStatusChange = async (index, row) => {
    await updateRecommendStatus(row.recommandStatus, [row.id]);
};
const handleResetSearch = () => {
    selectProductCateValue.value = [];
    listQuery.value = { pageNum: 1, pageSize: DEFAULT_PAGE_SIZE };
};
const handleDelete = async (index, row) => {
    updateDeleteStatus(1, [row.id]);
};
const handleUpdateProduct = (index, row) => {
    router.push({ path: '/pms/updateProduct', query: { id: row.id } });
};
const handleShowProduct = (index, row) => {
};
const handleShowVerifyDetail = (index, row) => {
};
const handleShowLog = (index, row) => {
};
const updatePublishStatus = async (publishStatus, ids) => {
    await productUpdatePublishStatusAPI({ ids: ids.join(','), publishStatus: publishStatus });
    ElMessage({
        message: t('message.modifySuccess'),
        type: 'success',
        duration: MESSAGE_DURATION_SHORT
    });
};
const updateNewStatus = async (newStatus, ids) => {
    await productUpdateNewStatusAPI({ ids: ids.join(','), newStatus: newStatus });
    ElMessage({
        message: t('message.modifySuccess'),
        type: 'success',
        duration: MESSAGE_DURATION_SHORT
    });
};
const updateRecommendStatus = async (recommendStatus, ids) => {
    await productUpdateRecommendStatusAPI({ ids: ids.join(','), recommendStatus: recommendStatus });
    ElMessage({
        message: t('message.modifySuccess'),
        type: 'success',
        duration: MESSAGE_DURATION_SHORT
    });
};
const updateDeleteStatus = async (deleteStatus, ids) => {
    await productUpdateDeleteStatusAPI({ ids: ids.join(','), deleteStatus: deleteStatus });
    ElMessage({
        message: t('message.deleteSuccess'),
        type: 'success',
        duration: MESSAGE_DURATION_SHORT
    });
    getList();
};
const verifyStatusFilter = (value) => {
    if (value === 1) {
        return t('product.verifyPassed');
    }
    else {
        return t('product.notVerified');
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
(__VLS_ctx.t('product.filterSearch'));
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
            [t, handleSearchList,];
        } });
const { default: __VLS_24 } = __VLS_20.slots;
(__VLS_ctx.t('product.queryResult'));
// @ts-ignore
[t,];
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
(__VLS_ctx.t('product.reset'));
// @ts-ignore
[t,];
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
    label: (__VLS_ctx.t('product.productName') + '：'),
}));
const __VLS_41 = __VLS_40({
    label: (__VLS_ctx.t('product.productName') + '：'),
}, ...__VLS_functionalComponentArgsRest(__VLS_40));
const { default: __VLS_44 } = __VLS_42.slots;
let __VLS_45;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_46 = __VLS_asFunctionalComponent1(__VLS_45, new __VLS_45({
    ...{ style: {} },
    modelValue: (__VLS_ctx.listQuery.keyword),
    placeholder: (__VLS_ctx.t('product.productName')),
}));
const __VLS_47 = __VLS_46({
    ...{ style: {} },
    modelValue: (__VLS_ctx.listQuery.keyword),
    placeholder: (__VLS_ctx.t('product.productName')),
}, ...__VLS_functionalComponentArgsRest(__VLS_46));
// @ts-ignore
[t, t, listQuery, listQuery,];
var __VLS_42;
let __VLS_50;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_51 = __VLS_asFunctionalComponent1(__VLS_50, new __VLS_50({
    label: (__VLS_ctx.t('product.productSn') + '：'),
}));
const __VLS_52 = __VLS_51({
    label: (__VLS_ctx.t('product.productSn') + '：'),
}, ...__VLS_functionalComponentArgsRest(__VLS_51));
const { default: __VLS_55 } = __VLS_53.slots;
let __VLS_56;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_57 = __VLS_asFunctionalComponent1(__VLS_56, new __VLS_56({
    ...{ style: {} },
    modelValue: (__VLS_ctx.listQuery.productSn),
    placeholder: (__VLS_ctx.t('product.productSn')),
}));
const __VLS_58 = __VLS_57({
    ...{ style: {} },
    modelValue: (__VLS_ctx.listQuery.productSn),
    placeholder: (__VLS_ctx.t('product.productSn')),
}, ...__VLS_functionalComponentArgsRest(__VLS_57));
// @ts-ignore
[t, t, listQuery,];
var __VLS_53;
let __VLS_61;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_62 = __VLS_asFunctionalComponent1(__VLS_61, new __VLS_61({
    label: (__VLS_ctx.t('product.productCategory') + '：'),
}));
const __VLS_63 = __VLS_62({
    label: (__VLS_ctx.t('product.productCategory') + '：'),
}, ...__VLS_functionalComponentArgsRest(__VLS_62));
const { default: __VLS_66 } = __VLS_64.slots;
let __VLS_67;
/** @ts-ignore @type { | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader'] | typeof __VLS_components.elCascader | typeof __VLS_components.ElCascader | typeof __VLS_components['el-cascader']} */
elCascader;
// @ts-ignore
const __VLS_68 = __VLS_asFunctionalComponent1(__VLS_67, new __VLS_67({
    clearable: true,
    modelValue: (__VLS_ctx.selectProductCateValue),
    options: (__VLS_ctx.productCateOptions),
}));
const __VLS_69 = __VLS_68({
    clearable: true,
    modelValue: (__VLS_ctx.selectProductCateValue),
    options: (__VLS_ctx.productCateOptions),
}, ...__VLS_functionalComponentArgsRest(__VLS_68));
// @ts-ignore
[t, selectProductCateValue, productCateOptions,];
var __VLS_64;
let __VLS_72;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_73 = __VLS_asFunctionalComponent1(__VLS_72, new __VLS_72({
    label: (__VLS_ctx.t('product.brand') + '：'),
}));
const __VLS_74 = __VLS_73({
    label: (__VLS_ctx.t('product.brand') + '：'),
}, ...__VLS_functionalComponentArgsRest(__VLS_73));
const { default: __VLS_77 } = __VLS_75.slots;
let __VLS_78;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_79 = __VLS_asFunctionalComponent1(__VLS_78, new __VLS_78({
    modelValue: (__VLS_ctx.listQuery.brandId),
    placeholder: (__VLS_ctx.t('product.selectBrand')),
    clearable: true,
    ...{ style: {} },
}));
const __VLS_80 = __VLS_79({
    modelValue: (__VLS_ctx.listQuery.brandId),
    placeholder: (__VLS_ctx.t('product.selectBrand')),
    clearable: true,
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_79));
const { default: __VLS_83 } = __VLS_81.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.brandOptions))) {
    let __VLS_84;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_85 = __VLS_asFunctionalComponent1(__VLS_84, new __VLS_84({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_86 = __VLS_85({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_85));
    // @ts-ignore
    [t, t, listQuery, brandOptions,];
}
// @ts-ignore
[];
var __VLS_81;
// @ts-ignore
[];
var __VLS_75;
let __VLS_89;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_90 = __VLS_asFunctionalComponent1(__VLS_89, new __VLS_89({
    label: (__VLS_ctx.t('product.onSale') + '：'),
}));
const __VLS_91 = __VLS_90({
    label: (__VLS_ctx.t('product.onSale') + '：'),
}, ...__VLS_functionalComponentArgsRest(__VLS_90));
const { default: __VLS_94 } = __VLS_92.slots;
let __VLS_95;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_96 = __VLS_asFunctionalComponent1(__VLS_95, new __VLS_95({
    modelValue: (__VLS_ctx.listQuery.publishStatus),
    placeholder: (__VLS_ctx.t('product.all')),
    clearable: true,
    ...{ style: {} },
}));
const __VLS_97 = __VLS_96({
    modelValue: (__VLS_ctx.listQuery.publishStatus),
    placeholder: (__VLS_ctx.t('product.all')),
    clearable: true,
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_96));
const { default: __VLS_100 } = __VLS_98.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.publishStatusOptions))) {
    let __VLS_101;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_102 = __VLS_asFunctionalComponent1(__VLS_101, new __VLS_101({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_103 = __VLS_102({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_102));
    // @ts-ignore
    [t, t, listQuery, publishStatusOptions,];
}
// @ts-ignore
[];
var __VLS_98;
// @ts-ignore
[];
var __VLS_92;
let __VLS_106;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_107 = __VLS_asFunctionalComponent1(__VLS_106, new __VLS_106({
    label: (__VLS_ctx.t('product.verifyStatus') + '：'),
}));
const __VLS_108 = __VLS_107({
    label: (__VLS_ctx.t('product.verifyStatus') + '：'),
}, ...__VLS_functionalComponentArgsRest(__VLS_107));
const { default: __VLS_111 } = __VLS_109.slots;
let __VLS_112;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_113 = __VLS_asFunctionalComponent1(__VLS_112, new __VLS_112({
    modelValue: (__VLS_ctx.listQuery.verifyStatus),
    placeholder: (__VLS_ctx.t('product.all')),
    clearable: true,
    ...{ style: {} },
}));
const __VLS_114 = __VLS_113({
    modelValue: (__VLS_ctx.listQuery.verifyStatus),
    placeholder: (__VLS_ctx.t('product.all')),
    clearable: true,
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_113));
const { default: __VLS_117 } = __VLS_115.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.verifyStatusOptions))) {
    let __VLS_118;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_119 = __VLS_asFunctionalComponent1(__VLS_118, new __VLS_118({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_120 = __VLS_119({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_119));
    // @ts-ignore
    [t, t, listQuery, verifyStatusOptions,];
}
// @ts-ignore
[];
var __VLS_115;
// @ts-ignore
[];
var __VLS_109;
// @ts-ignore
[];
var __VLS_36;
// @ts-ignore
[];
var __VLS_3;
let __VLS_123;
/** @ts-ignore @type { | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card'] | typeof __VLS_components.elCard | typeof __VLS_components.ElCard | typeof __VLS_components['el-card']} */
elCard;
// @ts-ignore
const __VLS_124 = __VLS_asFunctionalComponent1(__VLS_123, new __VLS_123({
    ...{ class: "operate-container" },
    shadow: "never",
}));
const __VLS_125 = __VLS_124({
    ...{ class: "operate-container" },
    shadow: "never",
}, ...__VLS_functionalComponentArgsRest(__VLS_124));
/** @type {__VLS_StyleScopedClasses['operate-container']} */ ;
const { default: __VLS_128 } = __VLS_126.slots;
let __VLS_129;
/** @ts-ignore @type { | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon'] | typeof __VLS_components.elIcon | typeof __VLS_components.ElIcon | typeof __VLS_components['el-icon']} */
elIcon;
// @ts-ignore
const __VLS_130 = __VLS_asFunctionalComponent1(__VLS_129, new __VLS_129({
    ...{ class: "el-icon-middle" },
}));
const __VLS_131 = __VLS_130({
    ...{ class: "el-icon-middle" },
}, ...__VLS_functionalComponentArgsRest(__VLS_130));
/** @type {__VLS_StyleScopedClasses['el-icon-middle']} */ ;
const { default: __VLS_134 } = __VLS_132.slots;
let __VLS_135;
/** @ts-ignore @type { | typeof __VLS_components.Tickets} */
Tickets;
// @ts-ignore
const __VLS_136 = __VLS_asFunctionalComponent1(__VLS_135, new __VLS_135({}));
const __VLS_137 = __VLS_136({}, ...__VLS_functionalComponentArgsRest(__VLS_136));
// @ts-ignore
[];
var __VLS_132;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
(__VLS_ctx.t('product.dataList'));
let __VLS_140;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_141 = __VLS_asFunctionalComponent1(__VLS_140, new __VLS_140({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}));
const __VLS_142 = __VLS_141({
    ...{ 'onClick': {} },
    ...{ class: "btn-add" },
}, ...__VLS_functionalComponentArgsRest(__VLS_141));
let __VLS_145;
const __VLS_146 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleAddProduct();
            // @ts-ignore
            [t, handleAddProduct,];
        } });
/** @type {__VLS_StyleScopedClasses['btn-add']} */ ;
const { default: __VLS_147 } = __VLS_143.slots;
(__VLS_ctx.t('common.add'));
// @ts-ignore
[t,];
var __VLS_143;
var __VLS_144;
// @ts-ignore
[];
var __VLS_126;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "table-container" },
});
/** @type {__VLS_StyleScopedClasses['table-container']} */ ;
let __VLS_148;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_149 = __VLS_asFunctionalComponent1(__VLS_148, new __VLS_148({
    ...{ 'onSelectionChange': {} },
    ref: "productTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}));
const __VLS_150 = __VLS_149({
    ...{ 'onSelectionChange': {} },
    ref: "productTable",
    data: (__VLS_ctx.list),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_149));
let __VLS_153;
const __VLS_154 = ({ selectionChange: {} },
    { onSelectionChange: (__VLS_ctx.handleSelectionChange) });
__VLS_asFunctionalDirective(__VLS_directives.vLoading, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.listLoading) }, null, null);
var __VLS_155 = {};
const { default: __VLS_157 } = __VLS_151.slots;
let __VLS_158;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_159 = __VLS_asFunctionalComponent1(__VLS_158, new __VLS_158({
    type: "selection",
    width: "60",
    align: "center",
}));
const __VLS_160 = __VLS_159({
    type: "selection",
    width: "60",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_159));
let __VLS_163;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_164 = __VLS_asFunctionalComponent1(__VLS_163, new __VLS_163({
    label: (__VLS_ctx.t('common.id')),
    width: "100",
    align: "center",
}));
const __VLS_165 = __VLS_164({
    label: (__VLS_ctx.t('common.id')),
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_164));
const { default: __VLS_168 } = __VLS_166.slots;
{
    const { default: __VLS_169 } = __VLS_166.slots;
    const [scope] = __VLS_vSlot(__VLS_169);
    (scope.row.id);
    // @ts-ignore
    [t, list, handleSelectionChange, vLoading, listLoading,];
}
// @ts-ignore
[];
var __VLS_166;
let __VLS_170;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_171 = __VLS_asFunctionalComponent1(__VLS_170, new __VLS_170({
    label: (__VLS_ctx.t('product.productImage')),
    width: "120",
    align: "center",
}));
const __VLS_172 = __VLS_171({
    label: (__VLS_ctx.t('product.productImage')),
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_171));
const { default: __VLS_175 } = __VLS_173.slots;
{
    const { default: __VLS_176 } = __VLS_173.slots;
    const [scope] = __VLS_vSlot(__VLS_176);
    __VLS_asFunctionalElement1(__VLS_intrinsics.img)({
        ...{ style: {} },
        src: (scope.row.pic),
        alt: (scope.row.name),
    });
    // @ts-ignore
    [t,];
}
// @ts-ignore
[];
var __VLS_173;
let __VLS_177;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_178 = __VLS_asFunctionalComponent1(__VLS_177, new __VLS_177({
    label: (__VLS_ctx.t('product.productName')),
    align: "center",
}));
const __VLS_179 = __VLS_178({
    label: (__VLS_ctx.t('product.productName')),
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_178));
const { default: __VLS_182 } = __VLS_180.slots;
{
    const { default: __VLS_183 } = __VLS_180.slots;
    const [scope] = __VLS_vSlot(__VLS_183);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    (scope.row.name);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    (__VLS_ctx.t('product.brand'));
    (scope.row.brandName);
    // @ts-ignore
    [t, t,];
}
// @ts-ignore
[];
var __VLS_180;
let __VLS_184;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_185 = __VLS_asFunctionalComponent1(__VLS_184, new __VLS_184({
    label: (__VLS_ctx.t('product.priceSn')),
    width: "120",
    align: "center",
}));
const __VLS_186 = __VLS_185({
    label: (__VLS_ctx.t('product.priceSn')),
    width: "120",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_185));
const { default: __VLS_189 } = __VLS_187.slots;
{
    const { default: __VLS_190 } = __VLS_187.slots;
    const [scope] = __VLS_vSlot(__VLS_190);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    (__VLS_ctx.t('product.price'));
    (scope.row.price);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    (__VLS_ctx.t('product.sn'));
    (scope.row.productSn);
    // @ts-ignore
    [t, t, t,];
}
// @ts-ignore
[];
var __VLS_187;
let __VLS_191;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_192 = __VLS_asFunctionalComponent1(__VLS_191, new __VLS_191({
    label: (__VLS_ctx.t('product.tag')),
    width: "140",
    align: "center",
}));
const __VLS_193 = __VLS_192({
    label: (__VLS_ctx.t('product.tag')),
    width: "140",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_192));
const { default: __VLS_196 } = __VLS_194.slots;
{
    const { default: __VLS_197 } = __VLS_194.slots;
    const [scope] = __VLS_vSlot(__VLS_197);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({
        ...{ style: {} },
    });
    (__VLS_ctx.t('product.onSale'));
    let __VLS_198;
    /** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
    elSwitch;
    // @ts-ignore
    const __VLS_199 = __VLS_asFunctionalComponent1(__VLS_198, new __VLS_198({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.publishStatus),
    }));
    const __VLS_200 = __VLS_199({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.publishStatus),
    }, ...__VLS_functionalComponentArgsRest(__VLS_199));
    let __VLS_203;
    const __VLS_204 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handlePublishStatusChange(scope.$index, scope.row);
                // @ts-ignore
                [t, t, handlePublishStatusChange,];
            } });
    var __VLS_201;
    var __VLS_202;
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({
        ...{ style: {} },
    });
    (__VLS_ctx.t('product.new'));
    let __VLS_205;
    /** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
    elSwitch;
    // @ts-ignore
    const __VLS_206 = __VLS_asFunctionalComponent1(__VLS_205, new __VLS_205({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.newStatus),
    }));
    const __VLS_207 = __VLS_206({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.newStatus),
    }, ...__VLS_functionalComponentArgsRest(__VLS_206));
    let __VLS_210;
    const __VLS_211 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handleNewStatusChange(scope.$index, scope.row);
                // @ts-ignore
                [t, handleNewStatusChange,];
            } });
    var __VLS_208;
    var __VLS_209;
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({
        ...{ style: {} },
    });
    (__VLS_ctx.t('product.recommended'));
    let __VLS_212;
    /** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
    elSwitch;
    // @ts-ignore
    const __VLS_213 = __VLS_asFunctionalComponent1(__VLS_212, new __VLS_212({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.recommandStatus),
    }));
    const __VLS_214 = __VLS_213({
        ...{ 'onChange': {} },
        activeValue: (1),
        inactiveValue: (0),
        modelValue: (scope.row.recommandStatus),
    }, ...__VLS_functionalComponentArgsRest(__VLS_213));
    let __VLS_217;
    const __VLS_218 = ({ change: {} },
        { onChange: (...[$event]) => {
                __VLS_ctx.handleRecommendStatusChange(scope.$index, scope.row);
                // @ts-ignore
                [t, handleRecommendStatusChange,];
            } });
    var __VLS_215;
    var __VLS_216;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_194;
let __VLS_219;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_220 = __VLS_asFunctionalComponent1(__VLS_219, new __VLS_219({
    label: (__VLS_ctx.t('product.sort')),
    width: "100",
    align: "center",
}));
const __VLS_221 = __VLS_220({
    label: (__VLS_ctx.t('product.sort')),
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_220));
const { default: __VLS_224 } = __VLS_222.slots;
{
    const { default: __VLS_225 } = __VLS_222.slots;
    const [scope] = __VLS_vSlot(__VLS_225);
    (scope.row.sort);
    // @ts-ignore
    [t,];
}
// @ts-ignore
[];
var __VLS_222;
let __VLS_226;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_227 = __VLS_asFunctionalComponent1(__VLS_226, new __VLS_226({
    label: (__VLS_ctx.t('product.skuStock')),
    width: "100",
    align: "center",
}));
const __VLS_228 = __VLS_227({
    label: (__VLS_ctx.t('product.skuStock')),
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_227));
const { default: __VLS_231 } = __VLS_229.slots;
{
    const { default: __VLS_232 } = __VLS_229.slots;
    const [scope] = __VLS_vSlot(__VLS_232);
    let __VLS_233;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_234 = __VLS_asFunctionalComponent1(__VLS_233, new __VLS_233({
        ...{ 'onClick': {} },
        type: "primary",
        icon: (__VLS_ctx.Edit),
        size: "large",
        circle: true,
    }));
    const __VLS_235 = __VLS_234({
        ...{ 'onClick': {} },
        type: "primary",
        icon: (__VLS_ctx.Edit),
        size: "large",
        circle: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_234));
    let __VLS_238;
    const __VLS_239 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleShowSkuEditDialog(scope.$index, scope.row);
                // @ts-ignore
                [t, Edit, handleShowSkuEditDialog,];
            } });
    var __VLS_236;
    var __VLS_237;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_229;
let __VLS_240;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_241 = __VLS_asFunctionalComponent1(__VLS_240, new __VLS_240({
    label: (__VLS_ctx.t('product.sales')),
    width: "100",
    align: "center",
}));
const __VLS_242 = __VLS_241({
    label: (__VLS_ctx.t('product.sales')),
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_241));
const { default: __VLS_245 } = __VLS_243.slots;
{
    const { default: __VLS_246 } = __VLS_243.slots;
    const [scope] = __VLS_vSlot(__VLS_246);
    (scope.row.sale);
    // @ts-ignore
    [t,];
}
// @ts-ignore
[];
var __VLS_243;
let __VLS_247;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_248 = __VLS_asFunctionalComponent1(__VLS_247, new __VLS_247({
    label: (__VLS_ctx.t('product.verifyStatus')),
    width: "100",
    align: "center",
}));
const __VLS_249 = __VLS_248({
    label: (__VLS_ctx.t('product.verifyStatus')),
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_248));
const { default: __VLS_252 } = __VLS_250.slots;
{
    const { default: __VLS_253 } = __VLS_250.slots;
    const [scope] = __VLS_vSlot(__VLS_253);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    (__VLS_ctx.verifyStatusFilter(scope.row.verifyStatus));
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    let __VLS_254;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_255 = __VLS_asFunctionalComponent1(__VLS_254, new __VLS_254({
        ...{ 'onClick': {} },
        type: "primary",
        link: true,
    }));
    const __VLS_256 = __VLS_255({
        ...{ 'onClick': {} },
        type: "primary",
        link: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_255));
    let __VLS_259;
    const __VLS_260 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleShowVerifyDetail(scope.$index, scope.row);
                // @ts-ignore
                [t, verifyStatusFilter, handleShowVerifyDetail,];
            } });
    const { default: __VLS_261 } = __VLS_257.slots;
    (__VLS_ctx.t('product.verifyDetail'));
    // @ts-ignore
    [t,];
    var __VLS_257;
    var __VLS_258;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_250;
let __VLS_262;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_263 = __VLS_asFunctionalComponent1(__VLS_262, new __VLS_262({
    label: (__VLS_ctx.t('product.operation')),
    width: "160",
    align: "center",
}));
const __VLS_264 = __VLS_263({
    label: (__VLS_ctx.t('product.operation')),
    width: "160",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_263));
const { default: __VLS_267 } = __VLS_265.slots;
{
    const { default: __VLS_268 } = __VLS_265.slots;
    const [scope] = __VLS_vSlot(__VLS_268);
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    let __VLS_269;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_270 = __VLS_asFunctionalComponent1(__VLS_269, new __VLS_269({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_271 = __VLS_270({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_270));
    let __VLS_274;
    const __VLS_275 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleShowProduct(scope.$index, scope.row);
                // @ts-ignore
                [t, handleShowProduct,];
            } });
    const { default: __VLS_276 } = __VLS_272.slots;
    (__VLS_ctx.t('product.view'));
    // @ts-ignore
    [t,];
    var __VLS_272;
    var __VLS_273;
    let __VLS_277;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_278 = __VLS_asFunctionalComponent1(__VLS_277, new __VLS_277({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_279 = __VLS_278({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_278));
    let __VLS_282;
    const __VLS_283 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleUpdateProduct(scope.$index, scope.row);
                // @ts-ignore
                [handleUpdateProduct,];
            } });
    const { default: __VLS_284 } = __VLS_280.slots;
    (__VLS_ctx.t('common.edit'));
    // @ts-ignore
    [t,];
    var __VLS_280;
    var __VLS_281;
    __VLS_asFunctionalElement1(__VLS_intrinsics.p, __VLS_intrinsics.p)({});
    let __VLS_285;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_286 = __VLS_asFunctionalComponent1(__VLS_285, new __VLS_285({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_287 = __VLS_286({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_286));
    let __VLS_290;
    const __VLS_291 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleShowLog(scope.$index, scope.row);
                // @ts-ignore
                [handleShowLog,];
            } });
    const { default: __VLS_292 } = __VLS_288.slots;
    (__VLS_ctx.t('product.log'));
    // @ts-ignore
    [t,];
    var __VLS_288;
    var __VLS_289;
    let __VLS_293;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_294 = __VLS_asFunctionalComponent1(__VLS_293, new __VLS_293({
        ...{ 'onClick': {} },
        size: "small",
        type: "danger",
    }));
    const __VLS_295 = __VLS_294({
        ...{ 'onClick': {} },
        size: "small",
        type: "danger",
    }, ...__VLS_functionalComponentArgsRest(__VLS_294));
    let __VLS_298;
    const __VLS_299 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleDelete(scope.$index, scope.row);
                // @ts-ignore
                [handleDelete,];
            } });
    const { default: __VLS_300 } = __VLS_296.slots;
    (__VLS_ctx.t('common.delete'));
    // @ts-ignore
    [t,];
    var __VLS_296;
    var __VLS_297;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_265;
// @ts-ignore
[];
var __VLS_151;
var __VLS_152;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "batch-operate-container" },
});
/** @type {__VLS_StyleScopedClasses['batch-operate-container']} */ ;
let __VLS_301;
/** @ts-ignore @type { | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select'] | typeof __VLS_components.elSelect | typeof __VLS_components.ElSelect | typeof __VLS_components['el-select']} */
elSelect;
// @ts-ignore
const __VLS_302 = __VLS_asFunctionalComponent1(__VLS_301, new __VLS_301({
    modelValue: (__VLS_ctx.operateType),
    placeholder: (__VLS_ctx.t('product.batchOperate')),
}));
const __VLS_303 = __VLS_302({
    modelValue: (__VLS_ctx.operateType),
    placeholder: (__VLS_ctx.t('product.batchOperate')),
}, ...__VLS_functionalComponentArgsRest(__VLS_302));
const { default: __VLS_306 } = __VLS_304.slots;
for (const [item] of __VLS_vFor((__VLS_ctx.operates))) {
    let __VLS_307;
    /** @ts-ignore @type { | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option'] | typeof __VLS_components.elOption | typeof __VLS_components.ElOption | typeof __VLS_components['el-option']} */
    elOption;
    // @ts-ignore
    const __VLS_308 = __VLS_asFunctionalComponent1(__VLS_307, new __VLS_307({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }));
    const __VLS_309 = __VLS_308({
        key: (item.value),
        label: (item.label),
        value: (item.value),
    }, ...__VLS_functionalComponentArgsRest(__VLS_308));
    // @ts-ignore
    [t, operateType, operates,];
}
// @ts-ignore
[];
var __VLS_304;
let __VLS_312;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_313 = __VLS_asFunctionalComponent1(__VLS_312, new __VLS_312({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}));
const __VLS_314 = __VLS_313({
    ...{ 'onClick': {} },
    ...{ style: {} },
    ...{ class: "search-button" },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_313));
let __VLS_317;
const __VLS_318 = ({ click: {} },
    { onClick: (...[$event]) => {
            __VLS_ctx.handleBatchOperate();
            // @ts-ignore
            [handleBatchOperate,];
        } });
/** @type {__VLS_StyleScopedClasses['search-button']} */ ;
const { default: __VLS_319 } = __VLS_315.slots;
(__VLS_ctx.t('common.confirm'));
// @ts-ignore
[t,];
var __VLS_315;
var __VLS_316;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "pagination-container" },
});
/** @type {__VLS_StyleScopedClasses['pagination-container']} */ ;
let __VLS_320;
/** @ts-ignore @type { | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination'] | typeof __VLS_components.elPagination | typeof __VLS_components.ElPagination | typeof __VLS_components['el-pagination']} */
elPagination;
// @ts-ignore
const __VLS_321 = __VLS_asFunctionalComponent1(__VLS_320, new __VLS_320({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: (__VLS_ctx.PAGE_SIZE_OPTIONS),
    currentPage: (__VLS_ctx.listQuery.pageNum),
    total: (__VLS_ctx.total),
}));
const __VLS_322 = __VLS_321({
    ...{ 'onSizeChange': {} },
    ...{ 'onCurrentChange': {} },
    background: true,
    layout: "total, sizes,prev, pager, next,jumper",
    pageSize: (__VLS_ctx.listQuery.pageSize),
    pageSizes: (__VLS_ctx.PAGE_SIZE_OPTIONS),
    currentPage: (__VLS_ctx.listQuery.pageNum),
    total: (__VLS_ctx.total),
}, ...__VLS_functionalComponentArgsRest(__VLS_321));
let __VLS_325;
const __VLS_326 = ({ sizeChange: {} },
    { onSizeChange: (__VLS_ctx.handleSizeChange) });
const __VLS_327 = ({ currentChange: {} },
    { onCurrentChange: (__VLS_ctx.handleCurrentChange) });
var __VLS_323;
var __VLS_324;
let __VLS_328;
/** @ts-ignore @type { | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog'] | typeof __VLS_components.elDialog | typeof __VLS_components.ElDialog | typeof __VLS_components['el-dialog']} */
elDialog;
// @ts-ignore
const __VLS_329 = __VLS_asFunctionalComponent1(__VLS_328, new __VLS_328({
    title: (__VLS_ctx.t('dialog.editProductInfo')),
    modelValue: (__VLS_ctx.editSkuInfo.dialogVisible),
    width: "40%",
}));
const __VLS_330 = __VLS_329({
    title: (__VLS_ctx.t('dialog.editProductInfo')),
    modelValue: (__VLS_ctx.editSkuInfo.dialogVisible),
    width: "40%",
}, ...__VLS_functionalComponentArgsRest(__VLS_329));
const { default: __VLS_333 } = __VLS_331.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
(__VLS_ctx.t('dialog.productSn'));
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({});
(__VLS_ctx.editSkuInfo.productSn);
let __VLS_334;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_335 = __VLS_asFunctionalComponent1(__VLS_334, new __VLS_334({
    placeholder: (__VLS_ctx.t('dialog.searchBySkuCode')),
    modelValue: (__VLS_ctx.editSkuInfo.keyword),
    ...{ style: {} },
}));
const __VLS_336 = __VLS_335({
    placeholder: (__VLS_ctx.t('dialog.searchBySkuCode')),
    modelValue: (__VLS_ctx.editSkuInfo.keyword),
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_335));
const { default: __VLS_339 } = __VLS_337.slots;
{
    const { append: __VLS_340 } = __VLS_337.slots;
    let __VLS_341;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_342 = __VLS_asFunctionalComponent1(__VLS_341, new __VLS_341({
        ...{ 'onClick': {} },
        icon: (__VLS_ctx.Search),
    }));
    const __VLS_343 = __VLS_342({
        ...{ 'onClick': {} },
        icon: (__VLS_ctx.Search),
    }, ...__VLS_functionalComponentArgsRest(__VLS_342));
    let __VLS_346;
    const __VLS_347 = ({ click: {} },
        { onClick: (__VLS_ctx.handleSearchEditSku) });
    var __VLS_344;
    var __VLS_345;
    // @ts-ignore
    [t, t, t, listQuery, listQuery, PAGE_SIZE_OPTIONS, total, handleSizeChange, handleCurrentChange, editSkuInfo, editSkuInfo, editSkuInfo, Search, handleSearchEditSku,];
}
// @ts-ignore
[];
var __VLS_337;
let __VLS_348;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_349 = __VLS_asFunctionalComponent1(__VLS_348, new __VLS_348({
    ...{ style: {} },
    data: (__VLS_ctx.editSkuInfo.stockList),
    border: true,
}));
const __VLS_350 = __VLS_349({
    ...{ style: {} },
    data: (__VLS_ctx.editSkuInfo.stockList),
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_349));
const { default: __VLS_353 } = __VLS_351.slots;
let __VLS_354;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_355 = __VLS_asFunctionalComponent1(__VLS_354, new __VLS_354({
    label: (__VLS_ctx.t('dialog.skuCode')),
    align: "center",
}));
const __VLS_356 = __VLS_355({
    label: (__VLS_ctx.t('dialog.skuCode')),
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_355));
const { default: __VLS_359 } = __VLS_357.slots;
{
    const { default: __VLS_360 } = __VLS_357.slots;
    const [scope] = __VLS_vSlot(__VLS_360);
    let __VLS_361;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_362 = __VLS_asFunctionalComponent1(__VLS_361, new __VLS_361({
        modelValue: (scope.row.skuCode),
    }));
    const __VLS_363 = __VLS_362({
        modelValue: (scope.row.skuCode),
    }, ...__VLS_functionalComponentArgsRest(__VLS_362));
    // @ts-ignore
    [t, editSkuInfo,];
}
// @ts-ignore
[];
var __VLS_357;
for (const [item, index] of __VLS_vFor((__VLS_ctx.editSkuInfo.productAttr))) {
    let __VLS_366;
    /** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
    elTableColumn;
    // @ts-ignore
    const __VLS_367 = __VLS_asFunctionalComponent1(__VLS_366, new __VLS_366({
        label: (item.name),
        key: (item.id),
        align: "center",
    }));
    const __VLS_368 = __VLS_367({
        label: (item.name),
        key: (item.id),
        align: "center",
    }, ...__VLS_functionalComponentArgsRest(__VLS_367));
    const { default: __VLS_371 } = __VLS_369.slots;
    {
        const { default: __VLS_372 } = __VLS_369.slots;
        const [scope] = __VLS_vSlot(__VLS_372);
        (__VLS_ctx.getProductSkuSp(scope.row, index));
        // @ts-ignore
        [editSkuInfo, getProductSkuSp,];
    }
    // @ts-ignore
    [];
    var __VLS_369;
    // @ts-ignore
    [];
}
let __VLS_373;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_374 = __VLS_asFunctionalComponent1(__VLS_373, new __VLS_373({
    label: (__VLS_ctx.t('dialog.salePrice')),
    width: "80",
    align: "center",
}));
const __VLS_375 = __VLS_374({
    label: (__VLS_ctx.t('dialog.salePrice')),
    width: "80",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_374));
const { default: __VLS_378 } = __VLS_376.slots;
{
    const { default: __VLS_379 } = __VLS_376.slots;
    const [scope] = __VLS_vSlot(__VLS_379);
    let __VLS_380;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_381 = __VLS_asFunctionalComponent1(__VLS_380, new __VLS_380({
        modelValue: (scope.row.price),
    }));
    const __VLS_382 = __VLS_381({
        modelValue: (scope.row.price),
    }, ...__VLS_functionalComponentArgsRest(__VLS_381));
    // @ts-ignore
    [t,];
}
// @ts-ignore
[];
var __VLS_376;
let __VLS_385;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_386 = __VLS_asFunctionalComponent1(__VLS_385, new __VLS_385({
    label: (__VLS_ctx.t('dialog.productStock')),
    width: "80",
    align: "center",
}));
const __VLS_387 = __VLS_386({
    label: (__VLS_ctx.t('dialog.productStock')),
    width: "80",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_386));
const { default: __VLS_390 } = __VLS_388.slots;
{
    const { default: __VLS_391 } = __VLS_388.slots;
    const [scope] = __VLS_vSlot(__VLS_391);
    let __VLS_392;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_393 = __VLS_asFunctionalComponent1(__VLS_392, new __VLS_392({
        modelValue: (scope.row.stock),
    }));
    const __VLS_394 = __VLS_393({
        modelValue: (scope.row.stock),
    }, ...__VLS_functionalComponentArgsRest(__VLS_393));
    // @ts-ignore
    [t,];
}
// @ts-ignore
[];
var __VLS_388;
let __VLS_397;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_398 = __VLS_asFunctionalComponent1(__VLS_397, new __VLS_397({
    label: (__VLS_ctx.t('dialog.lowStockWarning')),
    width: "100",
    align: "center",
}));
const __VLS_399 = __VLS_398({
    label: (__VLS_ctx.t('dialog.lowStockWarning')),
    width: "100",
    align: "center",
}, ...__VLS_functionalComponentArgsRest(__VLS_398));
const { default: __VLS_402 } = __VLS_400.slots;
{
    const { default: __VLS_403 } = __VLS_400.slots;
    const [scope] = __VLS_vSlot(__VLS_403);
    let __VLS_404;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_405 = __VLS_asFunctionalComponent1(__VLS_404, new __VLS_404({
        modelValue: (scope.row.lowStock),
    }));
    const __VLS_406 = __VLS_405({
        modelValue: (scope.row.lowStock),
    }, ...__VLS_functionalComponentArgsRest(__VLS_405));
    // @ts-ignore
    [t,];
}
// @ts-ignore
[];
var __VLS_400;
// @ts-ignore
[];
var __VLS_351;
{
    const { footer: __VLS_409 } = __VLS_331.slots;
    __VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
        ...{ class: "dialog-footer" },
    });
    /** @type {__VLS_StyleScopedClasses['dialog-footer']} */ ;
    let __VLS_410;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_411 = __VLS_asFunctionalComponent1(__VLS_410, new __VLS_410({
        ...{ 'onClick': {} },
    }));
    const __VLS_412 = __VLS_411({
        ...{ 'onClick': {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_411));
    let __VLS_415;
    const __VLS_416 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.editSkuInfo.dialogVisible = false;
                // @ts-ignore
                [editSkuInfo,];
            } });
    const { default: __VLS_417 } = __VLS_413.slots;
    (__VLS_ctx.t('common.cancel'));
    // @ts-ignore
    [t,];
    var __VLS_413;
    var __VLS_414;
    let __VLS_418;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_419 = __VLS_asFunctionalComponent1(__VLS_418, new __VLS_418({
        ...{ 'onClick': {} },
        type: "primary",
    }));
    const __VLS_420 = __VLS_419({
        ...{ 'onClick': {} },
        type: "primary",
    }, ...__VLS_functionalComponentArgsRest(__VLS_419));
    let __VLS_423;
    const __VLS_424 = ({ click: {} },
        { onClick: (__VLS_ctx.handleEditSkuConfirm) });
    const { default: __VLS_425 } = __VLS_421.slots;
    (__VLS_ctx.t('common.confirm'));
    // @ts-ignore
    [t, handleEditSkuConfirm,];
    var __VLS_421;
    var __VLS_422;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_331;
// @ts-ignore
var __VLS_156 = __VLS_155;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
