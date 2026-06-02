<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh, FullScreen } from '@element-plus/icons-vue'
import { getRealTimeDashboardAPI, type RealTimeDashboardDTO } from '@/apis/dataAnalysis'

const loading = ref(false)
const dashboardData = ref<RealTimeDashboardDTO>({
  todaySales: 0,
  todayOrders: 0,
  todayVisitors: 0,
  conversionRate: 0,
  todayProfit: 0,
  pendingOrders: 0,
  lowStockAlerts: 0,
  pendingPayments: 0,
  hotProducts: [],
  salesByHour: [],
  orderStatusDistribution: {},
  categorySalesDistribution: {},
  weekOverWeekGrowth: 0,
  monthOverMonthGrowth: 0
})

let refreshTimer: ReturnType<typeof setInterval> | null = null
const REFRESH_INTERVAL = 30000 // 30 seconds

const getDashboardData = async () => {
  loading.value = true
  try {
    const res = await getRealTimeDashboardAPI()
    if (res.data) {
      dashboardData.value = res.data
    }
  } catch (error) {
    ElMessage.error('Failed to load dashboard data')
  } finally {
    loading.value = false
  }
}

const formatCurrency = (value: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(value)
}

const formatNumber = (value: number) => {
  return new Intl.NumberFormat('en-US').format(value)
}

const getGrowthClass = (value: number) => {
  return value >= 0 ? 'growth-positive' : 'growth-negative'
}

const getHourLabel = (hour: number) => {
  return `${hour.toString().padStart(2, '0')}:00`
}

const handleRefresh = () => {
  getDashboardData()
}

const handleFullScreen = () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen()
  } else {
    document.exitFullscreen()
  }
}

onMounted(() => {
  getDashboardData()
  // Auto refresh every 30 seconds
  refreshTimer = setInterval(() => {
    getDashboardData()
  }, REFRESH_INTERVAL)
})

onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
  }
})
</script>

<template>
  <div class="dashboard-container">
    <div class="dashboard-header">
      <h2>Real-Time Operations Dashboard</h2>
      <div class="header-actions">
        <el-button type="primary" :icon="Refresh" @click="handleRefresh" :loading="loading">
          Refresh
        </el-button>
        <el-button :icon="FullScreen" @click="handleFullScreen">
          Full Screen
        </el-button>
      </div>
    </div>

    <!-- Key Metrics Row -->
    <el-row :gutter="20" class="metrics-row">
      <el-col :span="6">
        <div class="metric-card">
          <div class="metric-label">Today's Sales</div>
          <div class="metric-value">{{ formatCurrency(dashboardData.todaySales) }}</div>
          <div :class="['metric-growth', getGrowthClass(dashboardData.weekOverWeekGrowth)]">
            {{ dashboardData.weekOverWeekGrowth >= 0 ? '+' : '' }}{{ dashboardData.weekOverWeekGrowth }}% vs last week
          </div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="metric-card">
          <div class="metric-label">Today's Orders</div>
          <div class="metric-value">{{ formatNumber(dashboardData.todayOrders) }}</div>
          <div class="metric-sub">Conversion: {{ dashboardData.conversionRate }}%</div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="metric-card">
          <div class="metric-label">Today's Visitors</div>
          <div class="metric-value">{{ formatNumber(dashboardData.todayVisitors) }}</div>
          <div class="metric-sub">Active users</div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="metric-card">
          <div class="metric-label">Today's Profit</div>
          <div class="metric-value profit">{{ formatCurrency(dashboardData.todayProfit) }}</div>
          <div class="metric-sub">Estimated 15% margin</div>
        </div>
      </el-col>
    </el-row>

    <!-- Alerts Row -->
    <el-row :gutter="20" class="alerts-row">
      <el-col :span="8">
        <div class="alert-card pending-orders">
          <div class="alert-icon">📦</div>
          <div class="alert-content">
            <div class="alert-value">{{ dashboardData.pendingOrders }}</div>
            <div class="alert-label">Pending Orders</div>
          </div>
        </div>
      </el-col>
      <el-col :span="8">
        <div class="alert-card low-stock">
          <div class="alert-icon">⚠️</div>
          <div class="alert-content">
            <div class="alert-value">{{ dashboardData.lowStockAlerts }}</div>
            <div class="alert-label">Low Stock Alerts</div>
          </div>
        </div>
      </el-col>
      <el-col :span="8">
        <div class="alert-card pending-payments">
          <div class="alert-icon">💳</div>
          <div class="alert-content">
            <div class="alert-value">{{ dashboardData.pendingPayments }}</div>
            <div class="alert-label">Pending Payments</div>
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- Charts Row -->
    <el-row :gutter="20" class="charts-row">
      <el-col :span="12">
        <div class="chart-card">
          <h3>Sales by Hour</h3>
          <div class="chart-placeholder">
            <div class="bar-chart">
              <div
                v-for="item in dashboardData.salesByHour"
                :key="item.hour"
                class="bar"
                :style="{ height: `${Math.min((item.sales / 2000) * 100, 100)}%` }"
                :title="`${getHourLabel(item.hour)}: ${formatCurrency(item.sales)}`"
              >
                <span class="bar-label">{{ getHourLabel(item.hour) }}</span>
              </div>
            </div>
          </div>
        </div>
      </el-col>
      <el-col :span="12">
        <div class="chart-card">
          <h3>Order Status Distribution</h3>
          <div class="chart-placeholder">
            <div class="status-bars">
              <div
                v-for="(count, status) in dashboardData.orderStatusDistribution"
                :key="status"
                class="status-bar-item"
              >
                <div class="status-label">{{ status }}</div>
                <div class="status-bar">
                  <div
                    class="status-bar-fill"
                    :style="{ width: `${(count / 1373) * 100}%` }"
                  ></div>
                </div>
                <div class="status-count">{{ formatNumber(count) }}</div>
              </div>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- Bottom Row -->
    <el-row :gutter="20" class="bottom-row">
      <el-col :span="12">
        <div class="chart-card">
          <h3>Hot Products</h3>
          <el-table :data="dashboardData.hotProducts" style="width: 100%" size="small">
            <el-table-column prop="productName" label="Product" />
            <el-table-column prop="salesCount" label="Sales" width="100" align="right" />
            <el-table-column prop="salesAmount" label="Revenue" width="120" align="right">
              <template #default="{ row }">
                {{ formatCurrency(row.salesAmount) }}
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-col>
      <el-col :span="12">
        <div class="chart-card">
          <h3>Category Sales Distribution</h3>
          <div class="category-list">
            <div
              v-for="(sales, category) in dashboardData.categorySalesDistribution"
              :key="category"
              class="category-item"
            >
              <div class="category-name">{{ category }}</div>
              <div class="category-bar">
                <div
                  class="category-bar-fill"
                  :style="{ width: `${(sales / 385000) * 100}%` }"
                ></div>
              </div>
              <div class="category-sales">{{ formatCurrency(sales) }}</div>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.dashboard-container {
  padding: 20px;
  background: #f5f7fa;
  min-height: 100vh;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.dashboard-header h2 {
  margin: 0;
  font-size: 24px;
  color: #303133;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.metrics-row {
  margin-bottom: 20px;
}

.metric-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.metric-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 8px;
}

.metric-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 8px;
}

.metric-value.profit {
  color: #67c23a;
}

.metric-growth {
  font-size: 12px;
}

.growth-positive {
  color: #67c23a;
}

.growth-negative {
  color: #f56c6c;
}

.metric-sub {
  font-size: 12px;
  color: #909399;
}

.alerts-row {
  margin-bottom: 20px;
}

.alert-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.alert-icon {
  font-size: 32px;
  margin-right: 16px;
}

.alert-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
}

.alert-label {
  font-size: 14px;
  color: #909399;
}

.pending-orders {
  border-left: 4px solid #409eff;
}

.low-stock {
  border-left: 4px solid #e6a23c;
}

.pending-payments {
  border-left: 4px solid #67c23a;
}

.charts-row {
  margin-bottom: 20px;
}

.chart-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.chart-card h3 {
  margin: 0 0 16px 0;
  font-size: 16px;
  color: #303133;
}

.chart-placeholder {
  min-height: 200px;
}

.bar-chart {
  display: flex;
  align-items: flex-end;
  height: 200px;
  gap: 4px;
}

.bar {
  flex: 1;
  background: linear-gradient(to top, #409eff, #66b1ff);
  border-radius: 4px 4px 0 0;
  min-width: 20px;
  position: relative;
}

.bar-label {
  position: absolute;
  bottom: -20px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 10px;
  color: #909399;
  white-space: nowrap;
}

.status-bars {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.status-bar-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-label {
  width: 100px;
  font-size: 12px;
  color: #606266;
}

.status-bar {
  flex: 1;
  height: 20px;
  background: #f5f7fa;
  border-radius: 4px;
  overflow: hidden;
}

.status-bar-fill {
  height: 100%;
  background: linear-gradient(to right, #409eff, #66b1ff);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.status-count {
  width: 60px;
  text-align: right;
  font-size: 14px;
  font-weight: bold;
  color: #303133;
}

.bottom-row .chart-card {
  height: 320px;
  overflow: auto;
}

.category-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.category-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.category-name {
  width: 120px;
  font-size: 14px;
  color: #606266;
}

.category-bar {
  flex: 1;
  height: 24px;
  background: #f5f7fa;
  border-radius: 4px;
  overflow: hidden;
}

.category-bar-fill {
  height: 100%;
  background: linear-gradient(to right, #67c23a, #85ce61);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.category-sales {
  width: 100px;
  text-align: right;
  font-size: 14px;
  font-weight: bold;
  color: #303133;
}
</style>