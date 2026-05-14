/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/template-helpers.d.ts" />
/// <reference types="../../../../../../.npm/_npx/2db181330ea4b15b/node_modules/@vue/language-core/types/props-fallback.d.ts" />
import { ref, onMounted, computed } from 'vue';
import { str2Date } from '@/utils/datetime';
import img_home_order from '@/assets/images/home_order.png';
import img_home_today_amount from '@/assets/images/home_today_amount.png';
import img_home_yesterday_amount from '@/assets/images/home_yesterday_amount.png';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { LineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, LegendComponent, TitleComponent } from 'echarts/components';
// 通过use()方法按需注入ECharts的模块
use([
    CanvasRenderer, // 画布渲染器
    LineChart, // 折线图的绘制功能
    GridComponent, // 直角坐标系网格组件
    TooltipComponent, // 鼠标悬停时显示数据详情
    LegendComponent, // 图例组件
    TitleComponent // 显示图表标题
]);
// 默认图表数据
const defaultLineChartData = [
    { date: '2026-01-01', orderCount: 10, orderAmount: 1093 },
    { date: '2026-01-02', orderCount: 20, orderAmount: 2230 },
    { date: '2026-01-03', orderCount: 33, orderAmount: 3623 },
    { date: '2026-01-04', orderCount: 50, orderAmount: 6423 },
    { date: '2026-01-05', orderCount: 80, orderAmount: 8492 },
    { date: '2026-01-06', orderCount: 60, orderAmount: 6293 },
    { date: '2026-01-07', orderCount: 20, orderAmount: 2293 },
    { date: '2026-01-08', orderCount: 60, orderAmount: 6293 },
    { date: '2026-01-09', orderCount: 50, orderAmount: 5293 },
    { date: '2026-01-10', orderCount: 30, orderAmount: 3293 },
    { date: '2026-01-11', orderCount: 20, orderAmount: 2293 },
    { date: '2026-01-12', orderCount: 80, orderAmount: 8293 },
    { date: '2026-01-13', orderCount: 100, orderAmount: 10293 },
    { date: '2026-01-14', orderCount: 10, orderAmount: 1293 },
    { date: '2026-01-15', orderCount: 40, orderAmount: 4293 }
];
// 默认起始日期
const defaultStartDate = new Date(2026, 0, 1);
// 日期选择器日期范围[start,end]
const datePickerRange = ref([]);
// 初始化日期选择器数据
const initDatePickerRange = () => {
    const start = defaultStartDate;
    const end = new Date(start.getTime() + 1000 * 60 * 60 * 24 * 7);
    datePickerRange.value = [start, end];
};
// 图表数据
const lineChartData = ref([]);
// 图表数据加载状态
const loading = ref(false);
// 获取图表数据
const getLineChartData = () => {
    loading.value = true;
    setTimeout(() => {
        const start = datePickerRange.value[0];
        const end = datePickerRange.value[1];
        // 获取在当前区间范围内的数据
        lineChartData.value = defaultLineChartData.filter(item => {
            const currDate = str2Date(item.date);
            return currDate.getTime() >= start.getTime() && currDate.getTime() <= end.getTime();
        });
        loading.value = false;
    }, 1000);
};
// 组件挂载成功初始化数据
onMounted(() => {
    initDatePickerRange();
    getLineChartData();
});
// 日期选择器选项
const shortcuts = [
    {
        text: '最近一周',
        value: () => {
            const start = defaultStartDate;
            const end = new Date(start.getTime() + 1000 * 60 * 60 * 24 * 7);
            return [start, end];
        }
    },
    {
        text: '最近一月',
        value: () => {
            const start = defaultStartDate;
            const end = new Date(start.getTime() + 1000 * 60 * 60 * 24 * 30);
            return [start, end];
        }
    }
];
// 处理日期范围变化
const handleDatePickerRangeChange = () => {
    getLineChartData();
};
// X 轴：日期（2026-01-01 到 2026-01-15）
// 左 Y 轴：订单数量（0-100）
// 右 Y 轴：订单金额（0-10000+）
// 蓝色曲线：订单数量趋势（带填充）
// 绿色曲线：订单金额趋势（带填充）
// 鼠标悬停：显示交叉线和详细数据
// vue-charts中的选项
const chartOption = computed(() => {
    const dates = lineChartData.value.map(item => item.date);
    const orderCounts = lineChartData.value.map(item => item.orderCount);
    const orderAmounts = lineChartData.value.map(item => item.orderAmount);
    return {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross'
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: dates,
            axisLabel: {
                formatter: '{value}',
                rotate: 0
            }
        },
        yAxis: [
            {
                type: 'value',
                name: '订单数量',
                position: 'left',
                axisLabel: {
                    formatter: '{value}'
                }
            },
            {
                type: 'value',
                name: '订单金额',
                position: 'right',
                axisLabel: {
                    formatter: '{value}'
                }
            }
        ],
        series: [
            {
                name: '订单数量',
                type: 'line',
                areaStyle: {},
                data: orderCounts,
                smooth: true,
                itemStyle: {
                    color: '#409EFF'
                }
            },
            {
                name: '订单金额',
                type: 'line',
                yAxisIndex: 1,
                areaStyle: {},
                data: orderAmounts,
                smooth: true,
                itemStyle: {
                    color: '#67C23A'
                }
            }
        ]
    };
});
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
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-layout" },
});
/** @type {__VLS_StyleScopedClasses['total-layout']} */ ;
let __VLS_0;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent1(__VLS_0, new __VLS_0({
    gutter: (20),
}));
const __VLS_2 = __VLS_1({
    gutter: (20),
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
const { default: __VLS_5 } = __VLS_3.slots;
let __VLS_6;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent1(__VLS_6, new __VLS_6({
    span: (6),
}));
const __VLS_8 = __VLS_7({
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
const { default: __VLS_11 } = __VLS_9.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-frame" },
});
/** @type {__VLS_StyleScopedClasses['total-frame']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.img, __VLS_intrinsics.img)({
    src: (__VLS_ctx.img_home_order),
    ...{ class: "total-icon" },
});
/** @type {__VLS_StyleScopedClasses['total-icon']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-title" },
});
/** @type {__VLS_StyleScopedClasses['total-title']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-value" },
});
/** @type {__VLS_StyleScopedClasses['total-value']} */ ;
// @ts-ignore
[img_home_order,];
var __VLS_9;
let __VLS_12;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_13 = __VLS_asFunctionalComponent1(__VLS_12, new __VLS_12({
    span: (6),
}));
const __VLS_14 = __VLS_13({
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_13));
const { default: __VLS_17 } = __VLS_15.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-frame" },
});
/** @type {__VLS_StyleScopedClasses['total-frame']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.img, __VLS_intrinsics.img)({
    src: (__VLS_ctx.img_home_today_amount),
    ...{ class: "total-icon" },
});
/** @type {__VLS_StyleScopedClasses['total-icon']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-title" },
});
/** @type {__VLS_StyleScopedClasses['total-title']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-value" },
});
/** @type {__VLS_StyleScopedClasses['total-value']} */ ;
// @ts-ignore
[img_home_today_amount,];
var __VLS_15;
let __VLS_18;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_19 = __VLS_asFunctionalComponent1(__VLS_18, new __VLS_18({
    span: (6),
}));
const __VLS_20 = __VLS_19({
    span: (6),
}, ...__VLS_functionalComponentArgsRest(__VLS_19));
const { default: __VLS_23 } = __VLS_21.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-frame" },
});
/** @type {__VLS_StyleScopedClasses['total-frame']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.img, __VLS_intrinsics.img)({
    src: (__VLS_ctx.img_home_yesterday_amount),
    ...{ class: "total-icon" },
});
/** @type {__VLS_StyleScopedClasses['total-icon']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-title" },
});
/** @type {__VLS_StyleScopedClasses['total-title']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "total-value" },
});
/** @type {__VLS_StyleScopedClasses['total-value']} */ ;
// @ts-ignore
[img_home_yesterday_amount,];
var __VLS_21;
// @ts-ignore
[];
var __VLS_3;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-layout" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-layout']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "layout-title" },
});
/** @type {__VLS_StyleScopedClasses['layout-title']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-content" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-content']} */ ;
let __VLS_24;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_25 = __VLS_asFunctionalComponent1(__VLS_24, new __VLS_24({
    gutter: (20),
}));
const __VLS_26 = __VLS_25({
    gutter: (20),
}, ...__VLS_functionalComponentArgsRest(__VLS_25));
const { default: __VLS_29 } = __VLS_27.slots;
let __VLS_30;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_31 = __VLS_asFunctionalComponent1(__VLS_30, new __VLS_30({
    span: (8),
}));
const __VLS_32 = __VLS_31({
    span: (8),
}, ...__VLS_functionalComponentArgsRest(__VLS_31));
const { default: __VLS_35 } = __VLS_33.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-item" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-item']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
// @ts-ignore
[];
var __VLS_33;
let __VLS_36;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_37 = __VLS_asFunctionalComponent1(__VLS_36, new __VLS_36({
    span: (8),
}));
const __VLS_38 = __VLS_37({
    span: (8),
}, ...__VLS_functionalComponentArgsRest(__VLS_37));
const { default: __VLS_41 } = __VLS_39.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-item" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-item']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
// @ts-ignore
[];
var __VLS_39;
let __VLS_42;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_43 = __VLS_asFunctionalComponent1(__VLS_42, new __VLS_42({
    span: (8),
}));
const __VLS_44 = __VLS_43({
    span: (8),
}, ...__VLS_functionalComponentArgsRest(__VLS_43));
const { default: __VLS_47 } = __VLS_45.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-item" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-item']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
// @ts-ignore
[];
var __VLS_45;
// @ts-ignore
[];
var __VLS_27;
let __VLS_48;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_49 = __VLS_asFunctionalComponent1(__VLS_48, new __VLS_48({
    gutter: (20),
}));
const __VLS_50 = __VLS_49({
    gutter: (20),
}, ...__VLS_functionalComponentArgsRest(__VLS_49));
const { default: __VLS_53 } = __VLS_51.slots;
let __VLS_54;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent1(__VLS_54, new __VLS_54({
    span: (8),
}));
const __VLS_56 = __VLS_55({
    span: (8),
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
const { default: __VLS_59 } = __VLS_57.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-item" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-item']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
// @ts-ignore
[];
var __VLS_57;
let __VLS_60;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent1(__VLS_60, new __VLS_60({
    span: (8),
}));
const __VLS_62 = __VLS_61({
    span: (8),
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
const { default: __VLS_65 } = __VLS_63.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-item" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-item']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
// @ts-ignore
[];
var __VLS_63;
let __VLS_66;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_67 = __VLS_asFunctionalComponent1(__VLS_66, new __VLS_66({
    span: (8),
}));
const __VLS_68 = __VLS_67({
    span: (8),
}, ...__VLS_functionalComponentArgsRest(__VLS_67));
const { default: __VLS_71 } = __VLS_69.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-item" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-item']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
// @ts-ignore
[];
var __VLS_69;
// @ts-ignore
[];
var __VLS_51;
let __VLS_72;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_73 = __VLS_asFunctionalComponent1(__VLS_72, new __VLS_72({
    gutter: (20),
}));
const __VLS_74 = __VLS_73({
    gutter: (20),
}, ...__VLS_functionalComponentArgsRest(__VLS_73));
const { default: __VLS_77 } = __VLS_75.slots;
let __VLS_78;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_79 = __VLS_asFunctionalComponent1(__VLS_78, new __VLS_78({
    span: (8),
}));
const __VLS_80 = __VLS_79({
    span: (8),
}, ...__VLS_functionalComponentArgsRest(__VLS_79));
const { default: __VLS_83 } = __VLS_81.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-item" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-item']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
// @ts-ignore
[];
var __VLS_81;
let __VLS_84;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_85 = __VLS_asFunctionalComponent1(__VLS_84, new __VLS_84({
    span: (8),
}));
const __VLS_86 = __VLS_85({
    span: (8),
}, ...__VLS_functionalComponentArgsRest(__VLS_85));
const { default: __VLS_89 } = __VLS_87.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-item" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-item']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
// @ts-ignore
[];
var __VLS_87;
let __VLS_90;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_91 = __VLS_asFunctionalComponent1(__VLS_90, new __VLS_90({
    span: (8),
}));
const __VLS_92 = __VLS_91({
    span: (8),
}, ...__VLS_functionalComponentArgsRest(__VLS_91));
const { default: __VLS_95 } = __VLS_93.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "un-handle-item" },
});
/** @type {__VLS_StyleScopedClasses['un-handle-item']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "font-medium" },
});
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
    ...{ class: "color-danger" },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
// @ts-ignore
[];
var __VLS_93;
// @ts-ignore
[];
var __VLS_75;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "overview-layout" },
});
/** @type {__VLS_StyleScopedClasses['overview-layout']} */ ;
let __VLS_96;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_97 = __VLS_asFunctionalComponent1(__VLS_96, new __VLS_96({
    gutter: (20),
}));
const __VLS_98 = __VLS_97({
    gutter: (20),
}, ...__VLS_functionalComponentArgsRest(__VLS_97));
const { default: __VLS_101 } = __VLS_99.slots;
let __VLS_102;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_103 = __VLS_asFunctionalComponent1(__VLS_102, new __VLS_102({
    span: (12),
}));
const __VLS_104 = __VLS_103({
    span: (12),
}, ...__VLS_functionalComponentArgsRest(__VLS_103));
const { default: __VLS_107 } = __VLS_105.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "out-border" },
});
/** @type {__VLS_StyleScopedClasses['out-border']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "layout-title" },
});
/** @type {__VLS_StyleScopedClasses['layout-title']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_108;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_109 = __VLS_asFunctionalComponent1(__VLS_108, new __VLS_108({}));
const __VLS_110 = __VLS_109({}, ...__VLS_functionalComponentArgsRest(__VLS_109));
const { default: __VLS_113 } = __VLS_111.slots;
let __VLS_114;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_115 = __VLS_asFunctionalComponent1(__VLS_114, new __VLS_114({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}));
const __VLS_116 = __VLS_115({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}, ...__VLS_functionalComponentArgsRest(__VLS_115));
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
/** @type {__VLS_StyleScopedClasses['overview-item-value']} */ ;
const { default: __VLS_119 } = __VLS_117.slots;
// @ts-ignore
[];
var __VLS_117;
let __VLS_120;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_121 = __VLS_asFunctionalComponent1(__VLS_120, new __VLS_120({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}));
const __VLS_122 = __VLS_121({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}, ...__VLS_functionalComponentArgsRest(__VLS_121));
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
/** @type {__VLS_StyleScopedClasses['overview-item-value']} */ ;
const { default: __VLS_125 } = __VLS_123.slots;
// @ts-ignore
[];
var __VLS_123;
let __VLS_126;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_127 = __VLS_asFunctionalComponent1(__VLS_126, new __VLS_126({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}));
const __VLS_128 = __VLS_127({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}, ...__VLS_functionalComponentArgsRest(__VLS_127));
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
/** @type {__VLS_StyleScopedClasses['overview-item-value']} */ ;
const { default: __VLS_131 } = __VLS_129.slots;
// @ts-ignore
[];
var __VLS_129;
let __VLS_132;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_133 = __VLS_asFunctionalComponent1(__VLS_132, new __VLS_132({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}));
const __VLS_134 = __VLS_133({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}, ...__VLS_functionalComponentArgsRest(__VLS_133));
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
/** @type {__VLS_StyleScopedClasses['overview-item-value']} */ ;
const { default: __VLS_137 } = __VLS_135.slots;
// @ts-ignore
[];
var __VLS_135;
// @ts-ignore
[];
var __VLS_111;
let __VLS_138;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_139 = __VLS_asFunctionalComponent1(__VLS_138, new __VLS_138({
    ...{ class: "font-medium" },
}));
const __VLS_140 = __VLS_139({
    ...{ class: "font-medium" },
}, ...__VLS_functionalComponentArgsRest(__VLS_139));
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
const { default: __VLS_143 } = __VLS_141.slots;
let __VLS_144;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_145 = __VLS_asFunctionalComponent1(__VLS_144, new __VLS_144({
    span: (6),
    ...{ class: "overview-item-title" },
}));
const __VLS_146 = __VLS_145({
    span: (6),
    ...{ class: "overview-item-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_145));
/** @type {__VLS_StyleScopedClasses['overview-item-title']} */ ;
const { default: __VLS_149 } = __VLS_147.slots;
// @ts-ignore
[];
var __VLS_147;
let __VLS_150;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_151 = __VLS_asFunctionalComponent1(__VLS_150, new __VLS_150({
    span: (6),
    ...{ class: "overview-item-title" },
}));
const __VLS_152 = __VLS_151({
    span: (6),
    ...{ class: "overview-item-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_151));
/** @type {__VLS_StyleScopedClasses['overview-item-title']} */ ;
const { default: __VLS_155 } = __VLS_153.slots;
// @ts-ignore
[];
var __VLS_153;
let __VLS_156;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_157 = __VLS_asFunctionalComponent1(__VLS_156, new __VLS_156({
    span: (6),
    ...{ class: "overview-item-title" },
}));
const __VLS_158 = __VLS_157({
    span: (6),
    ...{ class: "overview-item-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_157));
/** @type {__VLS_StyleScopedClasses['overview-item-title']} */ ;
const { default: __VLS_161 } = __VLS_159.slots;
// @ts-ignore
[];
var __VLS_159;
let __VLS_162;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_163 = __VLS_asFunctionalComponent1(__VLS_162, new __VLS_162({
    span: (6),
    ...{ class: "overview-item-title" },
}));
const __VLS_164 = __VLS_163({
    span: (6),
    ...{ class: "overview-item-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_163));
/** @type {__VLS_StyleScopedClasses['overview-item-title']} */ ;
const { default: __VLS_167 } = __VLS_165.slots;
// @ts-ignore
[];
var __VLS_165;
// @ts-ignore
[];
var __VLS_141;
// @ts-ignore
[];
var __VLS_105;
let __VLS_168;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_169 = __VLS_asFunctionalComponent1(__VLS_168, new __VLS_168({
    span: (12),
}));
const __VLS_170 = __VLS_169({
    span: (12),
}, ...__VLS_functionalComponentArgsRest(__VLS_169));
const { default: __VLS_173 } = __VLS_171.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "out-border" },
});
/** @type {__VLS_StyleScopedClasses['out-border']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "layout-title" },
});
/** @type {__VLS_StyleScopedClasses['layout-title']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_174;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_175 = __VLS_asFunctionalComponent1(__VLS_174, new __VLS_174({}));
const __VLS_176 = __VLS_175({}, ...__VLS_functionalComponentArgsRest(__VLS_175));
const { default: __VLS_179 } = __VLS_177.slots;
let __VLS_180;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_181 = __VLS_asFunctionalComponent1(__VLS_180, new __VLS_180({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}));
const __VLS_182 = __VLS_181({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}, ...__VLS_functionalComponentArgsRest(__VLS_181));
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
/** @type {__VLS_StyleScopedClasses['overview-item-value']} */ ;
const { default: __VLS_185 } = __VLS_183.slots;
// @ts-ignore
[];
var __VLS_183;
let __VLS_186;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_187 = __VLS_asFunctionalComponent1(__VLS_186, new __VLS_186({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}));
const __VLS_188 = __VLS_187({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}, ...__VLS_functionalComponentArgsRest(__VLS_187));
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
/** @type {__VLS_StyleScopedClasses['overview-item-value']} */ ;
const { default: __VLS_191 } = __VLS_189.slots;
// @ts-ignore
[];
var __VLS_189;
let __VLS_192;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_193 = __VLS_asFunctionalComponent1(__VLS_192, new __VLS_192({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}));
const __VLS_194 = __VLS_193({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}, ...__VLS_functionalComponentArgsRest(__VLS_193));
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
/** @type {__VLS_StyleScopedClasses['overview-item-value']} */ ;
const { default: __VLS_197 } = __VLS_195.slots;
// @ts-ignore
[];
var __VLS_195;
let __VLS_198;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_199 = __VLS_asFunctionalComponent1(__VLS_198, new __VLS_198({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}));
const __VLS_200 = __VLS_199({
    span: (6),
    ...{ class: "color-danger overview-item-value" },
}, ...__VLS_functionalComponentArgsRest(__VLS_199));
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
/** @type {__VLS_StyleScopedClasses['overview-item-value']} */ ;
const { default: __VLS_203 } = __VLS_201.slots;
// @ts-ignore
[];
var __VLS_201;
// @ts-ignore
[];
var __VLS_177;
let __VLS_204;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_205 = __VLS_asFunctionalComponent1(__VLS_204, new __VLS_204({
    ...{ class: "font-medium" },
}));
const __VLS_206 = __VLS_205({
    ...{ class: "font-medium" },
}, ...__VLS_functionalComponentArgsRest(__VLS_205));
/** @type {__VLS_StyleScopedClasses['font-medium']} */ ;
const { default: __VLS_209 } = __VLS_207.slots;
let __VLS_210;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_211 = __VLS_asFunctionalComponent1(__VLS_210, new __VLS_210({
    span: (6),
    ...{ class: "overview-item-title" },
}));
const __VLS_212 = __VLS_211({
    span: (6),
    ...{ class: "overview-item-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_211));
/** @type {__VLS_StyleScopedClasses['overview-item-title']} */ ;
const { default: __VLS_215 } = __VLS_213.slots;
// @ts-ignore
[];
var __VLS_213;
let __VLS_216;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_217 = __VLS_asFunctionalComponent1(__VLS_216, new __VLS_216({
    span: (6),
    ...{ class: "overview-item-title" },
}));
const __VLS_218 = __VLS_217({
    span: (6),
    ...{ class: "overview-item-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_217));
/** @type {__VLS_StyleScopedClasses['overview-item-title']} */ ;
const { default: __VLS_221 } = __VLS_219.slots;
// @ts-ignore
[];
var __VLS_219;
let __VLS_222;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_223 = __VLS_asFunctionalComponent1(__VLS_222, new __VLS_222({
    span: (6),
    ...{ class: "overview-item-title" },
}));
const __VLS_224 = __VLS_223({
    span: (6),
    ...{ class: "overview-item-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_223));
/** @type {__VLS_StyleScopedClasses['overview-item-title']} */ ;
const { default: __VLS_227 } = __VLS_225.slots;
// @ts-ignore
[];
var __VLS_225;
let __VLS_228;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_229 = __VLS_asFunctionalComponent1(__VLS_228, new __VLS_228({
    span: (6),
    ...{ class: "overview-item-title" },
}));
const __VLS_230 = __VLS_229({
    span: (6),
    ...{ class: "overview-item-title" },
}, ...__VLS_functionalComponentArgsRest(__VLS_229));
/** @type {__VLS_StyleScopedClasses['overview-item-title']} */ ;
const { default: __VLS_233 } = __VLS_231.slots;
// @ts-ignore
[];
var __VLS_231;
// @ts-ignore
[];
var __VLS_207;
// @ts-ignore
[];
var __VLS_171;
// @ts-ignore
[];
var __VLS_99;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "statistics-layout" },
});
/** @type {__VLS_StyleScopedClasses['statistics-layout']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ class: "layout-title" },
});
/** @type {__VLS_StyleScopedClasses['layout-title']} */ ;
let __VLS_234;
/** @ts-ignore @type { | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row'] | typeof __VLS_components.elRow | typeof __VLS_components.ElRow | typeof __VLS_components['el-row']} */
elRow;
// @ts-ignore
const __VLS_235 = __VLS_asFunctionalComponent1(__VLS_234, new __VLS_234({}));
const __VLS_236 = __VLS_235({}, ...__VLS_functionalComponentArgsRest(__VLS_235));
const { default: __VLS_239 } = __VLS_237.slots;
let __VLS_240;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_241 = __VLS_asFunctionalComponent1(__VLS_240, new __VLS_240({
    span: (4),
}));
const __VLS_242 = __VLS_241({
    span: (4),
}, ...__VLS_functionalComponentArgsRest(__VLS_241));
const { default: __VLS_245 } = __VLS_243.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-success" },
    ...{ style: {} },
});
/** @type {__VLS_StyleScopedClasses['color-success']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-danger" },
    ...{ style: {} },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-success" },
    ...{ style: {} },
});
/** @type {__VLS_StyleScopedClasses['color-success']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({});
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ class: "color-danger" },
    ...{ style: {} },
});
/** @type {__VLS_StyleScopedClasses['color-danger']} */ ;
__VLS_asFunctionalElement1(__VLS_intrinsics.span, __VLS_intrinsics.span)({
    ...{ style: {} },
});
// @ts-ignore
[];
var __VLS_243;
let __VLS_246;
/** @ts-ignore @type { | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col'] | typeof __VLS_components.elCol | typeof __VLS_components.ElCol | typeof __VLS_components['el-col']} */
elCol;
// @ts-ignore
const __VLS_247 = __VLS_asFunctionalComponent1(__VLS_246, new __VLS_246({
    span: (20),
}));
const __VLS_248 = __VLS_247({
    span: (20),
}, ...__VLS_functionalComponentArgsRest(__VLS_247));
const { default: __VLS_251 } = __VLS_249.slots;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
let __VLS_252;
/** @ts-ignore @type { | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker'] | typeof __VLS_components.elDatePicker | typeof __VLS_components.ElDatePicker | typeof __VLS_components['el-date-picker']} */
elDatePicker;
// @ts-ignore
const __VLS_253 = __VLS_asFunctionalComponent1(__VLS_252, new __VLS_252({
    ...{ 'onChange': {} },
    ...{ style: {} },
    size: "small",
    modelValue: (__VLS_ctx.datePickerRange),
    type: "daterange",
    align: "right",
    unlinkPanels: true,
    rangeSeparator: "至",
    startPlaceholder: "开始日期",
    endPlaceholder: "结束日期",
    shortcuts: (__VLS_ctx.shortcuts),
}));
const __VLS_254 = __VLS_253({
    ...{ 'onChange': {} },
    ...{ style: {} },
    size: "small",
    modelValue: (__VLS_ctx.datePickerRange),
    type: "daterange",
    align: "right",
    unlinkPanels: true,
    rangeSeparator: "至",
    startPlaceholder: "开始日期",
    endPlaceholder: "结束日期",
    shortcuts: (__VLS_ctx.shortcuts),
}, ...__VLS_functionalComponentArgsRest(__VLS_253));
let __VLS_257;
const __VLS_258 = ({ change: {} },
    { onChange: (__VLS_ctx.handleDatePickerRangeChange) });
var __VLS_255;
var __VLS_256;
__VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
    ...{ style: {} },
});
if (!__VLS_ctx.loading) {
    let __VLS_259;
    /** @ts-ignore @type { | typeof __VLS_components.vChart | typeof __VLS_components.VChart | typeof __VLS_components['v-chart']} */
    vChart;
    // @ts-ignore
    const __VLS_260 = __VLS_asFunctionalComponent1(__VLS_259, new __VLS_259({
        option: (__VLS_ctx.chartOption),
        autoresize: true,
    }));
    const __VLS_261 = __VLS_260({
        option: (__VLS_ctx.chartOption),
        autoresize: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_260));
}
else {
    __VLS_asFunctionalElement1(__VLS_intrinsics.div, __VLS_intrinsics.div)({
        ...{ style: {} },
    });
    let __VLS_264;
    /** @ts-ignore @type { | typeof __VLS_components.elSkeleton | typeof __VLS_components.ElSkeleton | typeof __VLS_components['el-skeleton']} */
    elSkeleton;
    // @ts-ignore
    const __VLS_265 = __VLS_asFunctionalComponent1(__VLS_264, new __VLS_264({
        rows: (5),
        animated: true,
    }));
    const __VLS_266 = __VLS_265({
        rows: (5),
        animated: true,
    }, ...__VLS_functionalComponentArgsRest(__VLS_265));
}
// @ts-ignore
[datePickerRange, shortcuts, handleDatePickerRangeChange, loading, chartOption,];
var __VLS_249;
// @ts-ignore
[];
var __VLS_237;
// @ts-ignore
[];
const __VLS_export = (await import('vue')).defineComponent({});
export default {};
