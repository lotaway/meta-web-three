/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, computed, onMounted, inject } from 'vue';
import { ElMessage } from 'element-plus';
import { getMemberLevelListAPI } from '@/apis/memberLevel';
// 定义属性
const props = defineProps({
    isEdit: {
        type: Boolean,
        default: false
    }
});
// 定义事件
const emit = defineEmits(['prev-step', 'next-step']);
// 获取跨层传递的数据
const compProductParam = inject('product-key');
const getMemberLevelList = async () => {
    const res = await getMemberLevelListAPI({ defaultStatus: 0 });
    const memberPriceList = res.data.map(item => ({ memberLevelId: item.id, memberLevelName: item.name }));
    compProductParam.value.memberPriceList = memberPriceList;
};
// 组件挂载时调用
onMounted(() => {
    getMemberLevelList();
});
// 模板引用
const productSaleForm = ref();
// 日期禁用函数
const isDisabledDate = (time) => {
    return time.getTime() < Date.now();
};
// 被选中的服务保证，多个以逗号分割
const selectServiceList = computed({
    get() {
        if (!compProductParam.value.serviceIds)
            return [];
        return compProductParam.value.serviceIds.split(',').map(item => Number(item));
    },
    set(newValue) {
        let serviceIds = '';
        if (newValue && newValue.length > 0) {
            for (let i = 0; i < newValue.length; i++) {
                serviceIds += newValue[i] + ',';
            }
            if (serviceIds.endsWith(',')) {
                serviceIds = serviceIds.substring(0, serviceIds.length - 1);
            }
            compProductParam.value.serviceIds = serviceIds;
        }
        else {
            compProductParam.value.serviceIds = '';
        }
    }
});
// 处理删除阶梯价格操作
const handleRemoveProductLadder = (index) => {
    const productLadderList = compProductParam.value.productLadderList;
    if (!productLadderList)
        return;
    if (productLadderList.length === 1) {
        productLadderList.pop();
        productLadderList.push({
            count: 0,
            discount: 0,
            price: 0
        });
    }
    else {
        productLadderList.splice(index, 1);
    }
};
// 处理添加阶梯价格操作
const handleAddProductLadder = () => {
    const productLadderList = compProductParam.value.productLadderList;
    if (productLadderList && productLadderList.length < 3) {
        productLadderList.push({
            count: 0,
            discount: 0,
            price: 0
        });
    }
    else {
        ElMessage({
            message: '最多只能添加三条',
            type: 'warning'
        });
    }
};
// 处理删除满减价格操作
const handleRemoveFullReduction = (index) => {
    const fullReductionList = compProductParam.value.productFullReductionList;
    if (!fullReductionList)
        return;
    if (fullReductionList.length === 1) {
        fullReductionList.pop();
        fullReductionList.push({
            fullPrice: 0,
            reducePrice: 0
        });
    }
    else {
        fullReductionList.splice(index, 1);
    }
};
// 处理添加满减价格操作
const handleAddFullReduction = () => {
    const fullReductionList = compProductParam.value.productFullReductionList;
    if (fullReductionList && fullReductionList.length < 3) {
        fullReductionList.push({
            fullPrice: 0,
            reducePrice: 0
        });
    }
    else {
        ElMessage({
            message: '最多只能添加三条',
            type: 'warning'
        });
    }
};
// 处理上一步
const handlePrev = () => {
    emit('prev-step');
};
// 处理下一步
const handleNext = () => {
    emit('next-step');
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
    ...{ style: {} },
});
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form'] | typeof __VLS_components.elForm | typeof __VLS_components.ElForm | typeof __VLS_components['el-form']} */
elForm;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    model: (__VLS_ctx.compProductParam),
    ref: "productSaleForm",
    labelWidth: "120px",
    ...{ class: "form-inner-container" },
}));
const __VLS_2 = __VLS_1({
    model: (__VLS_ctx.compProductParam),
    ref: "productSaleForm",
    labelWidth: "120px",
    ...{ class: "form-inner-container" },
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_5 = {};
/** @type {__VLS_StyleScopedClasses['form-inner-container']} */ ;
const { default: __VLS_7 } = __VLS_3.slots;
let __VLS_8;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_9 = __VLS_asFunctionalComponent1(__VLS_8, new __VLS_8({
    label: "赠送积分：",
}));
const __VLS_10 = __VLS_9({
    label: "赠送积分：",
}, ...__VLS_functionalComponentArgsRest(__VLS_9));
const { default: __VLS_13 } = __VLS_11.slots;
let __VLS_14;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_15 = __VLS_asFunctionalComponent1(__VLS_14, new __VLS_14({
    modelValue: (__VLS_ctx.compProductParam.giftPoint),
}));
const __VLS_16 = __VLS_15({
    modelValue: (__VLS_ctx.compProductParam.giftPoint),
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
// @ts-ignore
[compProductParam, compProductParam,];
var __VLS_11;
let __VLS_19;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_20 = __VLS_asFunctionalComponent1(__VLS_19, new __VLS_19({
    label: "赠送成长值：",
}));
const __VLS_21 = __VLS_20({
    label: "赠送成长值：",
}, ...__VLS_functionalComponentArgsRest(__VLS_20));
const { default: __VLS_24 } = __VLS_22.slots;
let __VLS_25;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_26 = __VLS_asFunctionalComponent1(__VLS_25, new __VLS_25({
    modelValue: (__VLS_ctx.compProductParam.giftGrowth),
}));
const __VLS_27 = __VLS_26({
    modelValue: (__VLS_ctx.compProductParam.giftGrowth),
}, ...__VLS_functionalComponentArgsRest(__VLS_26));
// @ts-ignore
[compProductParam,];
var __VLS_22;
let __VLS_30;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_31 = __VLS_asFunctionalComponent1(__VLS_30, new __VLS_30({
    label: "积分购买限制：",
}));
const __VLS_32 = __VLS_31({
    label: "积分购买限制：",
}, ...__VLS_functionalComponentArgsRest(__VLS_31));
const { default: __VLS_35 } = __VLS_33.slots;
let __VLS_36;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_37 = __VLS_asFunctionalComponent1(__VLS_36, new __VLS_36({
    modelValue: (__VLS_ctx.compProductParam.usePointLimit),
}));
const __VLS_38 = __VLS_37({
    modelValue: (__VLS_ctx.compProductParam.usePointLimit),
}, ...__VLS_functionalComponentArgsRest(__VLS_37));
// @ts-ignore
[compProductParam,];
var __VLS_33;
let __VLS_41;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_42 = __VLS_asFunctionalComponent1(__VLS_41, new __VLS_41({
    label: "预告商品：",
}));
const __VLS_43 = __VLS_42({
    label: "预告商品：",
}, ...__VLS_functionalComponentArgsRest(__VLS_42));
const { default: __VLS_46 } = __VLS_44.slots;
let __VLS_47;
/** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
elSwitch;
// @ts-ignore
const __VLS_48 = __VLS_asFunctionalComponent1(__VLS_47, new __VLS_47({
    modelValue: (__VLS_ctx.compProductParam.previewStatus),
    activeValue: (1),
    inactiveValue: (0),
}));
const __VLS_49 = __VLS_48({
    modelValue: (__VLS_ctx.compProductParam.previewStatus),
    activeValue: (1),
    inactiveValue: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_48));
// @ts-ignore
[compProductParam,];
var __VLS_44;
let __VLS_52;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_53 = __VLS_asFunctionalComponent1(__VLS_52, new __VLS_52({
    label: "商品上架：",
}));
const __VLS_54 = __VLS_53({
    label: "商品上架：",
}, ...__VLS_functionalComponentArgsRest(__VLS_53));
const { default: __VLS_57 } = __VLS_55.slots;
let __VLS_58;
/** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
elSwitch;
// @ts-ignore
const __VLS_59 = __VLS_asFunctionalComponent1(__VLS_58, new __VLS_58({
    modelValue: (__VLS_ctx.compProductParam.publishStatus),
    activeValue: (1),
    inactiveValue: (0),
}));
const __VLS_60 = __VLS_59({
    modelValue: (__VLS_ctx.compProductParam.publishStatus),
    activeValue: (1),
    inactiveValue: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_59));
// @ts-ignore
[compProductParam,];
var __VLS_55;
let __VLS_63;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_64 = __VLS_asFunctionalComponent1(__VLS_63, new __VLS_63({
    label: "商品推荐：",
}));
const __VLS_65 = __VLS_64({
    label: "商品推荐：",
}, ...__VLS_functionalComponentArgsRest(__VLS_64));
const { default: __VLS_68 } = __VLS_66.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
});
let __VLS_69;
/** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
elSwitch;
// @ts-ignore
const __VLS_70 = __VLS_asFunctionalComponent1(__VLS_69, new __VLS_69({
    modelValue: (__VLS_ctx.compProductParam.newStatus),
    activeValue: (1),
    inactiveValue: (0),
}));
const __VLS_71 = __VLS_70({
    modelValue: (__VLS_ctx.compProductParam.newStatus),
    activeValue: (1),
    inactiveValue: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_70));
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
});
let __VLS_74;
/** @ts-ignore @type { | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch'] | typeof __VLS_components.elSwitch | typeof __VLS_components.ElSwitch | typeof __VLS_components['el-switch']} */
elSwitch;
// @ts-ignore
const __VLS_75 = __VLS_asFunctionalComponent1(__VLS_74, new __VLS_74({
    modelValue: (__VLS_ctx.compProductParam.recommandStatus),
    activeValue: (1),
    inactiveValue: (0),
}));
const __VLS_76 = __VLS_75({
    modelValue: (__VLS_ctx.compProductParam.recommandStatus),
    activeValue: (1),
    inactiveValue: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_75));
// @ts-ignore
[compProductParam, compProductParam,];
var __VLS_66;
let __VLS_79;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_80 = __VLS_asFunctionalComponent1(__VLS_79, new __VLS_79({
    label: "服务保证：",
}));
const __VLS_81 = __VLS_80({
    label: "服务保证：",
}, ...__VLS_functionalComponentArgsRest(__VLS_80));
const { default: __VLS_84 } = __VLS_82.slots;
let __VLS_85;
/** @ts-ignore @type { | typeof __VLS_components.elCheckboxGroup | typeof __VLS_components.ElCheckboxGroup | typeof __VLS_components['el-checkbox-group'] | typeof __VLS_components.elCheckboxGroup | typeof __VLS_components.ElCheckboxGroup | typeof __VLS_components['el-checkbox-group']} */
elCheckboxGroup;
// @ts-ignore
const __VLS_86 = __VLS_asFunctionalComponent1(__VLS_85, new __VLS_85({
    modelValue: (__VLS_ctx.selectServiceList),
}));
const __VLS_87 = __VLS_86({
    modelValue: (__VLS_ctx.selectServiceList),
}, ...__VLS_functionalComponentArgsRest(__VLS_86));
const { default: __VLS_90 } = __VLS_88.slots;
let __VLS_91;
/** @ts-ignore @type { | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox'] | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox']} */
elCheckbox;
// @ts-ignore
const __VLS_92 = __VLS_asFunctionalComponent1(__VLS_91, new __VLS_91({
    label: (1),
}));
const __VLS_93 = __VLS_92({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_92));
const { default: __VLS_96 } = __VLS_94.slots;
// @ts-ignore
[selectServiceList,];
var __VLS_94;
let __VLS_97;
/** @ts-ignore @type { | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox'] | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox']} */
elCheckbox;
// @ts-ignore
const __VLS_98 = __VLS_asFunctionalComponent1(__VLS_97, new __VLS_97({
    label: (2),
}));
const __VLS_99 = __VLS_98({
    label: (2),
}, ...__VLS_functionalComponentArgsRest(__VLS_98));
const { default: __VLS_102 } = __VLS_100.slots;
// @ts-ignore
[];
var __VLS_100;
let __VLS_103;
/** @ts-ignore @type { | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox'] | typeof __VLS_components.elCheckbox | typeof __VLS_components.ElCheckbox | typeof __VLS_components['el-checkbox']} */
elCheckbox;
// @ts-ignore
const __VLS_104 = __VLS_asFunctionalComponent1(__VLS_103, new __VLS_103({
    label: (3),
}));
const __VLS_105 = __VLS_104({
    label: (3),
}, ...__VLS_functionalComponentArgsRest(__VLS_104));
const { default: __VLS_108 } = __VLS_106.slots;
// @ts-ignore
[];
var __VLS_106;
// @ts-ignore
[];
var __VLS_88;
// @ts-ignore
[];
var __VLS_82;
let __VLS_109;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_110 = __VLS_asFunctionalComponent1(__VLS_109, new __VLS_109({
    label: "详细页标题：",
}));
const __VLS_111 = __VLS_110({
    label: "详细页标题：",
}, ...__VLS_functionalComponentArgsRest(__VLS_110));
const { default: __VLS_114 } = __VLS_112.slots;
let __VLS_115;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_116 = __VLS_asFunctionalComponent1(__VLS_115, new __VLS_115({
    modelValue: (__VLS_ctx.compProductParam.detailTitle),
}));
const __VLS_117 = __VLS_116({
    modelValue: (__VLS_ctx.compProductParam.detailTitle),
}, ...__VLS_functionalComponentArgsRest(__VLS_116));
// @ts-ignore
[compProductParam,];
var __VLS_112;
let __VLS_120;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_121 = __VLS_asFunctionalComponent1(__VLS_120, new __VLS_120({
    label: "详细页描述：",
}));
const __VLS_122 = __VLS_121({
    label: "详细页描述：",
}, ...__VLS_functionalComponentArgsRest(__VLS_121));
const { default: __VLS_125 } = __VLS_123.slots;
let __VLS_126;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_127 = __VLS_asFunctionalComponent1(__VLS_126, new __VLS_126({
    modelValue: (__VLS_ctx.compProductParam.detailDesc),
}));
const __VLS_128 = __VLS_127({
    modelValue: (__VLS_ctx.compProductParam.detailDesc),
}, ...__VLS_functionalComponentArgsRest(__VLS_127));
// @ts-ignore
[compProductParam,];
var __VLS_123;
let __VLS_131;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_132 = __VLS_asFunctionalComponent1(__VLS_131, new __VLS_131({
    label: "商品关键字：",
}));
const __VLS_133 = __VLS_132({
    label: "商品关键字：",
}, ...__VLS_functionalComponentArgsRest(__VLS_132));
const { default: __VLS_136 } = __VLS_134.slots;
let __VLS_137;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_138 = __VLS_asFunctionalComponent1(__VLS_137, new __VLS_137({
    modelValue: (__VLS_ctx.compProductParam.keywords),
}));
const __VLS_139 = __VLS_138({
    modelValue: (__VLS_ctx.compProductParam.keywords),
}, ...__VLS_functionalComponentArgsRest(__VLS_138));
// @ts-ignore
[compProductParam,];
var __VLS_134;
let __VLS_142;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_143 = __VLS_asFunctionalComponent1(__VLS_142, new __VLS_142({
    label: "商品备注：",
}));
const __VLS_144 = __VLS_143({
    label: "商品备注：",
}, ...__VLS_functionalComponentArgsRest(__VLS_143));
const { default: __VLS_147 } = __VLS_145.slots;
let __VLS_148;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_149 = __VLS_asFunctionalComponent1(__VLS_148, new __VLS_148({
    modelValue: (__VLS_ctx.compProductParam.note),
    type: "textarea",
    autoSize: (true),
}));
const __VLS_150 = __VLS_149({
    modelValue: (__VLS_ctx.compProductParam.note),
    type: "textarea",
    autoSize: (true),
}, ...__VLS_functionalComponentArgsRest(__VLS_149));
// @ts-ignore
[compProductParam,];
var __VLS_145;
let __VLS_153;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_154 = __VLS_asFunctionalComponent1(__VLS_153, new __VLS_153({
    label: "选择优惠方式：",
}));
const __VLS_155 = __VLS_154({
    label: "选择优惠方式：",
}, ...__VLS_functionalComponentArgsRest(__VLS_154));
const { default: __VLS_158 } = __VLS_156.slots;
let __VLS_159;
/** @ts-ignore @type { | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group'] | typeof __VLS_components.elRadioGroup | typeof __VLS_components.ElRadioGroup | typeof __VLS_components['el-radio-group']} */
elRadioGroup;
// @ts-ignore
const __VLS_160 = __VLS_asFunctionalComponent1(__VLS_159, new __VLS_159({
    modelValue: (__VLS_ctx.compProductParam.promotionType),
}));
const __VLS_161 = __VLS_160({
    modelValue: (__VLS_ctx.compProductParam.promotionType),
}, ...__VLS_functionalComponentArgsRest(__VLS_160));
const { default: __VLS_164 } = __VLS_162.slots;
let __VLS_165;
/** @ts-ignore @type { | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button'] | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button']} */
elRadioButton;
// @ts-ignore
const __VLS_166 = __VLS_asFunctionalComponent1(__VLS_165, new __VLS_165({
    label: (0),
}));
const __VLS_167 = __VLS_166({
    label: (0),
}, ...__VLS_functionalComponentArgsRest(__VLS_166));
const { default: __VLS_170 } = __VLS_168.slots;
// @ts-ignore
[compProductParam,];
var __VLS_168;
let __VLS_171;
/** @ts-ignore @type { | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button'] | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button']} */
elRadioButton;
// @ts-ignore
const __VLS_172 = __VLS_asFunctionalComponent1(__VLS_171, new __VLS_171({
    label: (1),
}));
const __VLS_173 = __VLS_172({
    label: (1),
}, ...__VLS_functionalComponentArgsRest(__VLS_172));
const { default: __VLS_176 } = __VLS_174.slots;
// @ts-ignore
[];
var __VLS_174;
let __VLS_177;
/** @ts-ignore @type { | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button'] | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button']} */
elRadioButton;
// @ts-ignore
const __VLS_178 = __VLS_asFunctionalComponent1(__VLS_177, new __VLS_177({
    label: (2),
}));
const __VLS_179 = __VLS_178({
    label: (2),
}, ...__VLS_functionalComponentArgsRest(__VLS_178));
const { default: __VLS_182 } = __VLS_180.slots;
// @ts-ignore
[];
var __VLS_180;
let __VLS_183;
/** @ts-ignore @type { | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button'] | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button']} */
elRadioButton;
// @ts-ignore
const __VLS_184 = __VLS_asFunctionalComponent1(__VLS_183, new __VLS_183({
    label: (3),
}));
const __VLS_185 = __VLS_184({
    label: (3),
}, ...__VLS_functionalComponentArgsRest(__VLS_184));
const { default: __VLS_188 } = __VLS_186.slots;
// @ts-ignore
[];
var __VLS_186;
let __VLS_189;
/** @ts-ignore @type { | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button'] | typeof __VLS_components.elRadioButton | typeof __VLS_components.ElRadioButton | typeof __VLS_components['el-radio-button']} */
elRadioButton;
// @ts-ignore
const __VLS_190 = __VLS_asFunctionalComponent1(__VLS_189, new __VLS_189({
    label: (4),
}));
const __VLS_191 = __VLS_190({
    label: (4),
}, ...__VLS_functionalComponentArgsRest(__VLS_190));
const { default: __VLS_194 } = __VLS_192.slots;
// @ts-ignore
[];
var __VLS_192;
// @ts-ignore
[];
var __VLS_162;
// @ts-ignore
[];
var __VLS_156;
let __VLS_195;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_196 = __VLS_asFunctionalComponent1(__VLS_195, new __VLS_195({}));
const __VLS_197 = __VLS_196({}, ...__VLS_functionalComponentArgsRest(__VLS_196));
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.compProductParam.promotionType === 1) }, null, null);
const { default: __VLS_200 } = __VLS_198.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
let __VLS_201;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_202 = __VLS_asFunctionalComponent1(__VLS_201, new __VLS_201({
    modelValue: (__VLS_ctx.compProductParam.promotionStartTime),
    type: "datetime",
    disabledDate: (__VLS_ctx.isDisabledDate),
    placeholder: "选择开始时间",
}));
const __VLS_203 = __VLS_202({
    modelValue: (__VLS_ctx.compProductParam.promotionStartTime),
    type: "datetime",
    disabledDate: (__VLS_ctx.isDisabledDate),
    placeholder: "选择开始时间",
}, ...__VLS_functionalComponentArgsRest(__VLS_202));
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "littleMargin" },
});
/** @type {__VLS_StyleScopedClasses['littleMargin']} */ ;
let __VLS_206;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_207 = __VLS_asFunctionalComponent1(__VLS_206, new __VLS_206({
    modelValue: (__VLS_ctx.compProductParam.promotionEndTime),
    type: "datetime",
    disabledDate: (__VLS_ctx.isDisabledDate),
    placeholder: "选择结束时间",
}));
const __VLS_208 = __VLS_207({
    modelValue: (__VLS_ctx.compProductParam.promotionEndTime),
    type: "datetime",
    disabledDate: (__VLS_ctx.isDisabledDate),
    placeholder: "选择结束时间",
}, ...__VLS_functionalComponentArgsRest(__VLS_207));
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "littleMargin" },
});
/** @type {__VLS_StyleScopedClasses['littleMargin']} */ ;
let __VLS_211;
/** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
elInput;
// @ts-ignore
const __VLS_212 = __VLS_asFunctionalComponent1(__VLS_211, new __VLS_211({
    ...{ style: {} },
    modelValue: (__VLS_ctx.compProductParam.promotionPrice),
    placeholder: "输入促销价格",
}));
const __VLS_213 = __VLS_212({
    ...{ style: {} },
    modelValue: (__VLS_ctx.compProductParam.promotionPrice),
    placeholder: "输入促销价格",
}, ...__VLS_functionalComponentArgsRest(__VLS_212));
// @ts-ignore
[compProductParam, compProductParam, compProductParam, compProductParam, isDisabledDate, isDisabledDate,];
var __VLS_198;
let __VLS_216;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_217 = __VLS_asFunctionalComponent1(__VLS_216, new __VLS_216({}));
const __VLS_218 = __VLS_217({}, ...__VLS_functionalComponentArgsRest(__VLS_217));
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.compProductParam.promotionType === 2) }, null, null);
const { default: __VLS_221 } = __VLS_219.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
for (const [item, index] of __VLS_vFor((__VLS_ctx.compProductParam.memberPriceList))) {
    __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
        ...{ class: ({ littleMargin: index !== 0 }) },
        key: (item.id),
    });
    /** @type {__VLS_StyleScopedClasses['littleMargin']} */ ;
    (item.memberLevelName);
    let __VLS_222;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_223 = __VLS_asFunctionalComponent1(__VLS_222, new __VLS_222({
        modelValue: (item.memberPrice),
        ...{ style: {} },
    }));
    const __VLS_224 = __VLS_223({
        modelValue: (item.memberPrice),
        ...{ style: {} },
    }, ...__VLS_functionalComponentArgsRest(__VLS_223));
    // @ts-ignore
    [compProductParam, compProductParam,];
}
// @ts-ignore
[];
var __VLS_219;
let __VLS_227;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_228 = __VLS_asFunctionalComponent1(__VLS_227, new __VLS_227({}));
const __VLS_229 = __VLS_228({}, ...__VLS_functionalComponentArgsRest(__VLS_228));
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.compProductParam.promotionType === 3) }, null, null);
const { default: __VLS_232 } = __VLS_230.slots;
let __VLS_233;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_234 = __VLS_asFunctionalComponent1(__VLS_233, new __VLS_233({
    data: (__VLS_ctx.compProductParam.productLadderList),
    ...{ style: {} },
    border: true,
}));
const __VLS_235 = __VLS_234({
    data: (__VLS_ctx.compProductParam.productLadderList),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_234));
const { default: __VLS_238 } = __VLS_236.slots;
let __VLS_239;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_240 = __VLS_asFunctionalComponent1(__VLS_239, new __VLS_239({
    label: "数量",
    align: "center",
    width: "120",
}));
const __VLS_241 = __VLS_240({
    label: "数量",
    align: "center",
    width: "120",
}, ...__VLS_functionalComponentArgsRest(__VLS_240));
const { default: __VLS_244 } = __VLS_242.slots;
{
    const { default: __VLS_245 } = __VLS_242.slots;
    const [scope] = __VLS_vSlot(__VLS_245);
    let __VLS_246;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_247 = __VLS_asFunctionalComponent1(__VLS_246, new __VLS_246({
        modelValue: (scope.row.count),
    }));
    const __VLS_248 = __VLS_247({
        modelValue: (scope.row.count),
    }, ...__VLS_functionalComponentArgsRest(__VLS_247));
    // @ts-ignore
    [compProductParam, compProductParam,];
}
// @ts-ignore
[];
var __VLS_242;
let __VLS_251;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_252 = __VLS_asFunctionalComponent1(__VLS_251, new __VLS_251({
    label: "折扣",
    align: "center",
    width: "120",
}));
const __VLS_253 = __VLS_252({
    label: "折扣",
    align: "center",
    width: "120",
}, ...__VLS_functionalComponentArgsRest(__VLS_252));
const { default: __VLS_256 } = __VLS_254.slots;
{
    const { default: __VLS_257 } = __VLS_254.slots;
    const [scope] = __VLS_vSlot(__VLS_257);
    let __VLS_258;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_259 = __VLS_asFunctionalComponent1(__VLS_258, new __VLS_258({
        modelValue: (scope.row.discount),
    }));
    const __VLS_260 = __VLS_259({
        modelValue: (scope.row.discount),
    }, ...__VLS_functionalComponentArgsRest(__VLS_259));
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_254;
let __VLS_263;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_264 = __VLS_asFunctionalComponent1(__VLS_263, new __VLS_263({
    align: "center",
    label: "操作",
}));
const __VLS_265 = __VLS_264({
    align: "center",
    label: "操作",
}, ...__VLS_functionalComponentArgsRest(__VLS_264));
const { default: __VLS_268 } = __VLS_266.slots;
{
    const { default: __VLS_269 } = __VLS_266.slots;
    const [scope] = __VLS_vSlot(__VLS_269);
    let __VLS_270;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_271 = __VLS_asFunctionalComponent1(__VLS_270, new __VLS_270({
        ...{ 'onClick': {} },
        type: "text",
    }));
    const __VLS_272 = __VLS_271({
        ...{ 'onClick': {} },
        type: "text",
    }, ...__VLS_functionalComponentArgsRest(__VLS_271));
    let __VLS_275;
    const __VLS_276 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleRemoveProductLadder(scope.$index);
                // @ts-ignore
                [handleRemoveProductLadder,];
            } });
    const { default: __VLS_277 } = __VLS_273.slots;
    // @ts-ignore
    [];
    var __VLS_273;
    var __VLS_274;
    let __VLS_278;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_279 = __VLS_asFunctionalComponent1(__VLS_278, new __VLS_278({
        ...{ 'onClick': {} },
        type: "text",
    }));
    const __VLS_280 = __VLS_279({
        ...{ 'onClick': {} },
        type: "text",
    }, ...__VLS_functionalComponentArgsRest(__VLS_279));
    let __VLS_283;
    const __VLS_284 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleAddProductLadder();
                // @ts-ignore
                [handleAddProductLadder,];
            } });
    const { default: __VLS_285 } = __VLS_281.slots;
    // @ts-ignore
    [];
    var __VLS_281;
    var __VLS_282;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_266;
// @ts-ignore
[];
var __VLS_236;
// @ts-ignore
[];
var __VLS_230;
let __VLS_286;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_287 = __VLS_asFunctionalComponent1(__VLS_286, new __VLS_286({}));
const __VLS_288 = __VLS_287({}, ...__VLS_functionalComponentArgsRest(__VLS_287));
__VLS_asFunctionalDirective(__VLS_directives.vShow, {})(null, { ...__VLS_directiveBindingRestFields, value: (__VLS_ctx.compProductParam.promotionType === 4) }, null, null);
const { default: __VLS_291 } = __VLS_289.slots;
let __VLS_292;
/** @ts-ignore @type { | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table'] | typeof __VLS_components.elTable | typeof __VLS_components.ElTable | typeof __VLS_components['el-table']} */
elTable;
// @ts-ignore
const __VLS_293 = __VLS_asFunctionalComponent1(__VLS_292, new __VLS_292({
    data: (__VLS_ctx.compProductParam.productFullReductionList),
    ...{ style: {} },
    border: true,
}));
const __VLS_294 = __VLS_293({
    data: (__VLS_ctx.compProductParam.productFullReductionList),
    ...{ style: {} },
    border: true,
}, ...__VLS_functionalComponentArgsRest(__VLS_293));
const { default: __VLS_297 } = __VLS_295.slots;
let __VLS_298;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_299 = __VLS_asFunctionalComponent1(__VLS_298, new __VLS_298({
    label: "满",
    align: "center",
    width: "120",
}));
const __VLS_300 = __VLS_299({
    label: "满",
    align: "center",
    width: "120",
}, ...__VLS_functionalComponentArgsRest(__VLS_299));
const { default: __VLS_303 } = __VLS_301.slots;
{
    const { default: __VLS_304 } = __VLS_301.slots;
    const [scope] = __VLS_vSlot(__VLS_304);
    let __VLS_305;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_306 = __VLS_asFunctionalComponent1(__VLS_305, new __VLS_305({
        modelValue: (scope.row.fullPrice),
    }));
    const __VLS_307 = __VLS_306({
        modelValue: (scope.row.fullPrice),
    }, ...__VLS_functionalComponentArgsRest(__VLS_306));
    // @ts-ignore
    [compProductParam, compProductParam,];
}
// @ts-ignore
[];
var __VLS_301;
let __VLS_310;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_311 = __VLS_asFunctionalComponent1(__VLS_310, new __VLS_310({
    label: "立减",
    align: "center",
    width: "120",
}));
const __VLS_312 = __VLS_311({
    label: "立减",
    align: "center",
    width: "120",
}, ...__VLS_functionalComponentArgsRest(__VLS_311));
const { default: __VLS_315 } = __VLS_313.slots;
{
    const { default: __VLS_316 } = __VLS_313.slots;
    const [scope] = __VLS_vSlot(__VLS_316);
    let __VLS_317;
    /** @ts-ignore @type { | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input'] | typeof __VLS_components.elInput | typeof __VLS_components.ElInput | typeof __VLS_components['el-input']} */
    elInput;
    // @ts-ignore
    const __VLS_318 = __VLS_asFunctionalComponent1(__VLS_317, new __VLS_317({
        modelValue: (scope.row.reducePrice),
    }));
    const __VLS_319 = __VLS_318({
        modelValue: (scope.row.reducePrice),
    }, ...__VLS_functionalComponentArgsRest(__VLS_318));
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_313;
let __VLS_322;
/** @ts-ignore @type { | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column'] | typeof __VLS_components.elTableColumn | typeof __VLS_components.ElTableColumn | typeof __VLS_components['el-table-column']} */
elTableColumn;
// @ts-ignore
const __VLS_323 = __VLS_asFunctionalComponent1(__VLS_322, new __VLS_322({
    align: "center",
    label: "操作",
}));
const __VLS_324 = __VLS_323({
    align: "center",
    label: "操作",
}, ...__VLS_functionalComponentArgsRest(__VLS_323));
const { default: __VLS_327 } = __VLS_325.slots;
{
    const { default: __VLS_328 } = __VLS_325.slots;
    const [scope] = __VLS_vSlot(__VLS_328);
    let __VLS_329;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_330 = __VLS_asFunctionalComponent1(__VLS_329, new __VLS_329({
        ...{ 'onClick': {} },
        type: "text",
    }));
    const __VLS_331 = __VLS_330({
        ...{ 'onClick': {} },
        type: "text",
    }, ...__VLS_functionalComponentArgsRest(__VLS_330));
    let __VLS_334;
    const __VLS_335 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleRemoveFullReduction(scope.$index);
                // @ts-ignore
                [handleRemoveFullReduction,];
            } });
    const { default: __VLS_336 } = __VLS_332.slots;
    // @ts-ignore
    [];
    var __VLS_332;
    var __VLS_333;
    let __VLS_337;
    /** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
    elButton;
    // @ts-ignore
    const __VLS_338 = __VLS_asFunctionalComponent1(__VLS_337, new __VLS_337({
        ...{ 'onClick': {} },
        type: "text",
    }));
    const __VLS_339 = __VLS_338({
        ...{ 'onClick': {} },
        type: "text",
    }, ...__VLS_functionalComponentArgsRest(__VLS_338));
    let __VLS_342;
    const __VLS_343 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.handleAddFullReduction();
                // @ts-ignore
                [handleAddFullReduction,];
            } });
    const { default: __VLS_344 } = __VLS_340.slots;
    // @ts-ignore
    [];
    var __VLS_340;
    var __VLS_341;
    // @ts-ignore
    [];
}
// @ts-ignore
[];
var __VLS_325;
// @ts-ignore
[];
var __VLS_295;
// @ts-ignore
[];
var __VLS_289;
let __VLS_345;
/** @ts-ignore @type { | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item'] | typeof __VLS_components.elFormItem | typeof __VLS_components.ElFormItem | typeof __VLS_components['el-form-item']} */
elFormItem;
// @ts-ignore
const __VLS_346 = __VLS_asFunctionalComponent1(__VLS_345, new __VLS_345({}));
const __VLS_347 = __VLS_346({}, ...__VLS_functionalComponentArgsRest(__VLS_346));
const { default: __VLS_350 } = __VLS_348.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_351;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_352 = __VLS_asFunctionalComponent1(__VLS_351, new __VLS_351({
    ...{ 'onClick': {} },
}));
const __VLS_353 = __VLS_352({
    ...{ 'onClick': {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_352));
let __VLS_356;
const __VLS_357 = ({ click: {} },
    { onClick: (__VLS_ctx.handlePrev) });
const { default: __VLS_358 } = __VLS_354.slots;
// @ts-ignore
[handlePrev,];
var __VLS_354;
var __VLS_355;
let __VLS_359;
/** @ts-ignore @type { | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button'] | typeof __VLS_components.elButton | typeof __VLS_components.ElButton | typeof __VLS_components['el-button']} */
elButton;
// @ts-ignore
const __VLS_360 = __VLS_asFunctionalComponent1(__VLS_359, new __VLS_359({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_361 = __VLS_360({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_360));
let __VLS_364;
const __VLS_365 = ({ click: {} },
    { onClick: (__VLS_ctx.handleNext) });
const { default: __VLS_366 } = __VLS_362.slots;
// @ts-ignore
[handleNext,];
var __VLS_362;
var __VLS_363;
// @ts-ignore
[];
var __VLS_348;
// @ts-ignore
[];
var __VLS_3;
// @ts-ignore
var __VLS_6 = __VLS_5;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({
    emits: {},
    props: {
        isEdit: {
            type: Boolean,
            default: false
        }
    },
});
export default {};
