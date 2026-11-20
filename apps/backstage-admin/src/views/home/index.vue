<script setup lang="ts">
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
import { ref, onMounted, computed } from 'vue'
import { str2Date } from '@/utils/datetime'
import img_home_order from '@/assets/images/home_order.png'
import img_home_today_amount from '@/assets/images/home_today_amount.png'
import img_home_yesterday_amount from '@/assets/images/home_yesterday_amount.png'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import VChart from 'vue-echarts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent
} from 'echarts/components'

// Inject ECharts modules on demand via use()
use([
  CanvasRenderer, // Canvas renderer
  LineChart, // Line chart drawing capability
  GridComponent, // Cartesian grid component
  TooltipComponent, // Show data details on hover
  LegendComponent, // Legend component
  TitleComponent // Show chart title
])

// Line chart data type
type LineChartDataItem = {
  date: string, // Transaction date
  orderCount: number, // Order count
  orderAmount: number // Order amount
}

// Default chart data
const defaultLineChartData: LineChartDataItem[] = [
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
]

// Default start date
const defaultStartDate = new Date(2026, 0, 1)

// Date picker range [start, end]
const datePickerRange = ref<Date[]>([])
// Initialize date picker data
const initDatePickerRange = () => {
  const start = defaultStartDate
  const end = new Date(start.getTime() + 1000 * 60 * 60 * 24 * 7)
  datePickerRange.value = [start, end] as Date[]
}
// Chart data
const lineChartData = ref<LineChartDataItem[]>([])
// Chart data loading state
const loading = ref(false)
// Get chart data
const getLineChartData = () => {
  loading.value = true
  setTimeout(() => {
    const start = datePickerRange.value[0]
    const end = datePickerRange.value[1]
    // Get data within the current range
    lineChartData.value = defaultLineChartData.filter(item => {
      const currDate = str2Date(item.date)
      return currDate!.getTime() >= start!.getTime() && currDate!.getTime() <= end!.getTime()
    })
    loading.value = false
  }, 1000)
}

// Initialize data on successful mount
onMounted(() => {
  initDatePickerRange()
  getLineChartData()
})

// Date picker options
const shortcuts = [
  {
    text: '最近一周',
    value: () => {
      const start = defaultStartDate
      const end = new Date(start.getTime() + 1000 * 60 * 60 * 24 * 7)
      return [start, end]
    }
  },
  {
    text: '最近一月',
    value: () => {
      const start = defaultStartDate
      const end = new Date(start.getTime() + 1000 * 60 * 60 * 24 * 30)
      return [start, end]
    }
  }
]
// Handle date range change
const handleDatePickerRangeChange = () => {
  getLineChartData()
}

const buildChartBaseOption = () => ({
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
  }
})

const buildChartXAxis = (dates: string[]) => ({
  type: 'category',
  boundaryGap: false,
  data: dates,
  axisLabel: {
    formatter: '{value}',
    rotate: 0
  }
})

const buildChartYAxis = () => [
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
]

const buildChartSeries = (orderCounts: number[], orderAmounts: number[]) => [
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

// X axis: date (2026-01-01 to 2026-01-15)
// Left Y axis: order count (0-100)
// Right Y axis: order amount (0-10000+)
// Blue curve: order count trend (with fill)
// Green curve: order amount trend (with fill)
// Hover: show crosshair and detailed data
// Options in vue-charts
const chartOption = computed(() => {
  const dates = lineChartData.value.map(item => item.date)
  const orderCounts = lineChartData.value.map(item => item.orderCount)
  const orderAmounts = lineChartData.value.map(item => item.orderAmount)
  return {
    ...buildChartBaseOption(),
    xAxis: buildChartXAxis(dates),
    yAxis: buildChartYAxis(),
    series: buildChartSeries(orderCounts, orderAmounts)
  }
})
</script>

<template>
  <div class="app-container">
    <div class="total-layout">
      <el-row :gutter="20">
        <el-col :span="6">
          <div class="total-frame">
            <img :src="img_home_order" class="total-icon">
            <div class="total-title">{{ t('home.todayOrders') }}</div>
            <div class="total-value">200</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="total-frame">
            <img :src="img_home_today_amount" class="total-icon">
            <div class="total-title">{{ t('home.todaySales') }}</div>
            <div class="total-value">￥5000.00</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="total-frame">
            <img :src="img_home_yesterday_amount" class="total-icon">
            <div class="total-title">{{ t('home.yesterdaySales') }}</div>
            <div class="total-value">￥5000.00</div>
          </div>
        </el-col>
      </el-row>
    </div>

    <div class="un-handle-layout">
      <div class="layout-title">{{ t('home.pendingTasks') }}</div>
      <div class="un-handle-content">
        <el-row :gutter="20">
          <el-col :span="8">
            <div class="un-handle-item">
              <span class="font-medium">{{ t('home.unpaidOrders') }}</span>
              <span style="float: right" class="color-danger">(10)</span>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="un-handle-item">
              <span class="font-medium">{{ t('home.completedOrders') }}</span>
              <span style="float: right" class="color-danger">(10)</span>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="un-handle-item">
              <span class="font-medium">{{ t('home.awaitingReceiptOrders') }}</span>
              <span style="float: right" class="color-danger">(10)</span>
            </div>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="8">
            <div class="un-handle-item">
              <span class="font-medium">{{ t('home.unshippedOrders') }}</span>
              <span style="float: right" class="color-danger">(10)</span>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="un-handle-item">
              <span class="font-medium">{{ t('home.newOutOfStock') }}</span>
              <span style="float: right" class="color-danger">(10)</span>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="un-handle-item">
              <span class="font-medium">{{ t('home.pendingRefunds') }}</span>
              <span style="float: right" class="color-danger">(10)</span>
            </div>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="8">
            <div class="un-handle-item">
              <span class="font-medium">{{ t('home.shippedOrders') }}</span>
              <span style="float: right" class="color-danger">(10)</span>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="un-handle-item">
              <span class="font-medium">{{ t('home.pendingReturns') }}</span>
              <span style="float: right" class="color-danger">(10)</span>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="un-handle-item">
              <span class="font-medium">{{ t('home.expiringAds') }}</span>
              <span style="float: right" class="color-danger">(10)</span>
            </div>
          </el-col>
        </el-row>
      </div>
    </div>
    <div class="overview-layout">
      <el-row :gutter="20">
        <el-col :span="12">
          <div class="out-border">
            <div class="layout-title">{{ t('home.productOverview') }}</div>
            <div style="padding: 40px">
              <el-row>
                <el-col :span="6" class="color-danger overview-item-value">100</el-col>
                <el-col :span="6" class="color-danger overview-item-value">400</el-col>
                <el-col :span="6" class="color-danger overview-item-value">50</el-col>
                <el-col :span="6" class="color-danger overview-item-value">500</el-col>
              </el-row>
              <el-row class="font-medium">
                <el-col :span="6" class="overview-item-title">{{ t('home.delisted') }}</el-col>
                <el-col :span="6" class="overview-item-title">{{ t('home.listed') }}</el-col>
                <el-col :span="6" class="overview-item-title">{{ t('home.lowStock') }}</el-col>
                <el-col :span="6" class="overview-item-title">{{ t('home.allProducts') }}</el-col>
              </el-row>
            </div>
          </div>
        </el-col>
        <el-col :span="12">
          <div class="out-border">
            <div class="layout-title">{{ t('home.userOverview') }}</div>
            <div style="padding: 40px">
              <el-row>
                <el-col :span="6" class="color-danger overview-item-value">100</el-col>
                <el-col :span="6" class="color-danger overview-item-value">200</el-col>
                <el-col :span="6" class="color-danger overview-item-value">1000</el-col>
                <el-col :span="6" class="color-danger overview-item-value">5000</el-col>
              </el-row>
              <el-row class="font-medium">
                <el-col :span="6" class="overview-item-title">{{ t('home.todayNew') }}</el-col>
                <el-col :span="6" class="overview-item-title">{{ t('home.yesterdayNew') }}</el-col>
                <el-col :span="6" class="overview-item-title">{{ t('home.monthNew') }}</el-col>
                <el-col :span="6" class="overview-item-title">{{ t('home.totalMembers') }}</el-col>
              </el-row>
            </div>
          </div>
        </el-col>
      </el-row>
    </div>
    <div class="statistics-layout">
      <div class="layout-title">{{ t('home.orderStats') }}</div>
      <el-row>
        <el-col :span="4">
          <div style="padding: 20px">
            <div>
              <div style="color: #909399;font-size: 14px">{{ t('home.monthOrders') }}</div>
              <div style="color: #606266;font-size: 24px;padding: 10px 0">10000</div>
              <div>
                <span class="color-success" style="font-size: 14px">+10%</span>
                <span style="color: #C0C4CC;font-size: 14px">{{ t('home.vsLastMonth') }}</span>
              </div>
            </div>
            <div style="margin-top: 20px;">
              <div style="color: #909399;font-size: 14px">{{ t('home.weekOrders') }}</div>
              <div style="color: #606266;font-size: 24px;padding: 10px 0">1000</div>
              <div>
                <span class="color-danger" style="font-size: 14px">-10%</span>
                <span style="color: #C0C4CC;font-size: 14px">{{ t('home.vsLastWeek') }}</span>
              </div>
            </div>
            <div style="margin-top: 20px;">
              <div style="color: #909399;font-size: 14px">{{ t('home.monthSales') }}</div>
              <div style="color: #606266;font-size: 24px;padding: 10px 0">100000</div>
              <div>
                <span class="color-success" style="font-size: 14px">+10%</span>
                <span style="color: #C0C4CC;font-size: 14px">{{ t('home.vsLastMonth') }}</span>
              </div>
            </div>
            <div style="margin-top: 20px;">
              <div style="color: #909399;font-size: 14px">{{ t('home.weekSales') }}</div>
              <div style="color: #606266;font-size: 24px;padding: 10px 0">50000</div>
              <div>
                <span class="color-danger" style="font-size: 14px">-10%</span>
                <span style="color: #C0C4CC;font-size: 14px">{{ t('home.vsLastWeek') }}</span>
              </div>
            </div>
          </div>
        </el-col>
        <el-col :span="20">
          <div style="padding: 10px;border-left:1px solid #DCDFE6">
            <el-date-picker style="float: right;z-index: 1" size="small" v-model="datePickerRange" type="daterange"
              align="right" unlink-panels :range-separator="t('common.to')" :start-placeholder="t('common.startDate')" :end-placeholder="t('common.endDate')"
              :shortcuts="shortcuts" @change="handleDatePickerRangeChange">
            </el-date-picker>
            <div style="height: 400px;">
              <v-chart v-if="!loading" :option="chartOption" autoresize />
              <div v-else
                style="display: flex; justify-content: center; align-items: center; height: 100%;width: 100%;">
                <el-skeleton :rows="5" animated />
              </div>
            </div>
          </div>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<style scoped>
.app-container {
  margin-top: 40px;
  margin-left: 120px;
  margin-right: 120px;
}

.total-layout {
  margin-top: 20px;
}

.total-frame {
  border: 1px solid #DCDFE6;
  padding: 20px;
  height: 100px;
}

.total-icon {
  color: #409EFF;
  width: 60px;
  height: 60px;
}

.total-title {
  position: relative;
  font-size: 16px;
  color: #909399;
  left: 70px;
  top: -50px;
}

.total-value {
  position: relative;
  font-size: 18px;
  color: #606266;
  left: 70px;
  top: -40px;
}

.un-handle-layout {
  margin-top: 20px;
  border: 1px solid #DCDFE6;
}

.layout-title {
  color: #606266;
  padding: 15px 20px;
  background: #F2F6FC;
  font-weight: bold;
}

.un-handle-content {
  padding: 20px 40px;
}

.un-handle-item {
  border-bottom: 1px solid #EBEEF5;
  padding: 10px;
}

.overview-layout {
  margin-top: 20px;
}

.overview-item-value {
  font-size: 24px;
  text-align: center;
}

.overview-item-title {
  margin-top: 10px;
  text-align: center;
}

.out-border {
  border: 1px solid #DCDFE6;
}

.statistics-layout {
  margin-top: 20px;
  border: 1px solid #DCDFE6;
}

.mine-layout {
  position: absolute;
  right: 140px;
  top: 107px;
  width: 250px;
  height: 235px;
}

.address-content {
  padding: 20px;
  font-size: 18px
}
</style>
