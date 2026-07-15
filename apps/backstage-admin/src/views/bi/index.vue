<template>
  <div class="bi-dashboard">
    <div class="page-header">
      <h2>{{ t('bi.title') }}</h2>
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        :start-placeholder="t('bi.startDate')"
        :end-placeholder="t('bi.endDate')"
        value-format="YYYY-MM-DD"
        @change="loadAll"
      />
    </div>

    <el-tabs v-model="activeTab">
      <!-- Sales Analysis -->
      <el-tab-pane :label="t('bi.salesAnalysis')" name="sales">
        <el-row :gutter="20" class="stat-cards">
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatMoney(salesRevenue) }}</div><div class="stat-label">{{ t('bi.totalRevenue') }}</div></div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatMoney(salesCost) }}</div><div class="stat-label">{{ t('bi.totalCost') }}</div></div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatMoney(salesProfit) }}</div><div class="stat-label">{{ t('bi.grossProfit') }}</div></div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ salesOrders }}</div><div class="stat-label">{{ t('bi.totalOrders') }}</div></div></el-card></el-col>
        </el-row>
        <el-card class="section-card" v-loading="loading">
          <template #header><span>{{ t('bi.salesTrend') }}</span></template>
          <el-table :data="salesTrendRows" stripe size="small" empty-text="暂无数据">
            <el-table-column prop="date" label="日期" width="120" />
            <el-table-column prop="amount" label="销售额" width="150"><template #default="{ row }">{{ formatMoney(row.amount) }}</template></el-table-column>
            <el-table-column prop="orderCount" label="订单数" width="100" />
          </el-table>
        </el-card>
        <el-row :gutter="20" class="section-row">
          <el-col :span="12">
            <el-card v-loading="loading">
              <template #header><span>{{ t('bi.categoryDistribution') }}</span></template>
              <el-table :data="categoryList" stripe size="small" max-height="300" empty-text="暂无数据">
                <el-table-column prop="categoryName" :label="t('bi.categoryName')" />
                <el-table-column prop="salesAmount" :label="t('bi.salesAmount')" width="120"><template #default="{ row }">{{ formatMoney(row.salesAmount) }}</template></el-table-column>
                <el-table-column prop="growthRate" label="增长率" width="80"><template #default="{ row }">{{ (row.growthRate * 100).toFixed(1) }}%</template></el-table-column>
              </el-table>
            </el-card>
          </el-col>
          <el-col :span="12">
            <el-card v-loading="loading">
              <template #header><span>{{ t('bi.regionalComparison') }}</span></template>
              <el-table :data="regionList" stripe size="small" max-height="300" empty-text="暂无数据">
                <el-table-column prop="region" :label="t('bi.region')" />
                <el-table-column prop="salesAmount" :label="t('bi.salesAmount')" width="120"><template #default="{ row }">{{ formatMoney(row.salesAmount) }}</template></el-table-column>
                <el-table-column prop="orderCount" :label="t('bi.orderCount')" width="80" />
              </el-table>
            </el-card>
          </el-col>
        </el-row>
      </el-tab-pane>

      <!-- Financial Analysis -->
      <el-tab-pane :label="t('bi.financialAnalysis')" name="financial">
        <el-row :gutter="20" class="stat-cards">
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatMoney(financialRevenue) }}</div><div class="stat-label">{{ t('bi.revenue') }}</div></div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ financialOrders }}</div><div class="stat-label">{{ t('bi.totalOrders') }}</div></div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatMoney(financialProfit) }}</div><div class="stat-label">{{ t('bi.netProfit') }}</div></div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatPercent(financialMargin) }}</div><div class="stat-label">{{ t('bi.grossMargin') }}</div></div></el-card></el-col>
        </el-row>
      </el-tab-pane>

      <!-- Inventory Analysis -->
      <el-tab-pane :label="t('bi.inventoryAnalysis')" name="inventory">
        <el-row :gutter="20" class="stat-cards">
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ inventoryTotalProducts }}</div><div class="stat-label">{{ t('bi.totalProducts') }}</div></div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ inventoryTotalQty }}</div><div class="stat-label">{{ t('bi.totalQuantity') }}</div></div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ inventoryTotalValue }}</div><div class="stat-label">{{ t('bi.inventoryValue') }}</div></div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value warning">{{ inventoryLowStock }}</div><div class="stat-label">{{ t('bi.safetyStockAlert') }}</div></div></el-card></el-col>
        </el-row>
        <el-card class="section-card" v-loading="loading">
          <template #header><span>{{ t('bi.safetyStockAlert') }}</span></template>
          <el-table :data="safetyAlerts" stripe size="small" empty-text="暂无数据">
            <el-table-column prop="productName" :label="t('bi.productName')" />
            <el-table-column prop="warehouseName" :label="t('bi.warehouse')" />
            <el-table-column prop="quantity" :label="t('bi.currentStock')" width="100" />
            <el-table-column prop="minStock" :label="t('bi.safetyStock')" width="100" />
          </el-table>
        </el-card>
      </el-tab-pane>

      <!-- Production Analysis -->
      <el-tab-pane :label="t('bi.productionAnalysis')" name="production">
        <el-row :gutter="20" class="stat-cards">
          <el-col :span="8"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatPercent(productionYield) }}</div><div class="stat-label">{{ t('bi.yieldRate') }}</div></div></el-card></el-col>
          <el-col :span="8"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ productionOutput }}</div><div class="stat-label">{{ t('bi.productionOutput') }}</div></div></el-card></el-col>
          <el-col :span="8"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ productionDefects }}</div><div class="stat-label">{{ t('bi.defectCount') }}</div></div></el-card></el-col>
        </el-row>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { t } from '@/locales'
import {
  getSalesTrend, getCategoryDistribution, getRegionalComparison,
  getFinancialSummary, getSafetyStockAlerts, getSalesFunnel,
} from '@/apis/bi'
import type { CategoryDistribution, FinancialSummary } from '@/apis/bi'

const defaultStart = new Date(new Date().getFullYear(), 0, 1).toISOString().split('T')[0]
const defaultEnd = new Date().toISOString().split('T')[0]

const dateRange = ref<string[]>([defaultStart, defaultEnd])
const loading = ref(false)
const activeTab = ref('sales')

const salesTrendRows = ref<{ date: string; amount: number; orderCount: number }[]>([])
const categoryList = ref<CategoryDistribution[]>([])
const regionList = ref<{ region: string; salesAmount: number; orderCount: number }[]>([])
const financialData = ref<FinancialSummary>({ totalRevenue: 0, totalCost: 0, grossProfit: 0, netProfit: 0, orderCount: 0 })
const inventoryOverview = ref({ totalProducts: 0, totalQuantity: 0, lowStockCount: 0, totalValue: 0 })
const safetyAlerts = ref<any[]>([])

const salesRevenue = computed(() => salesTrendRows.value.reduce((s, r) => s + r.amount, 0))
const salesCost = computed(() => salesRevenue.value * 0.6)
const salesProfit = computed(() => salesRevenue.value - salesCost.value)
const salesOrders = computed(() => salesTrendRows.value.reduce((s, r) => s + r.orderCount, 0))
const financialRevenue = computed(() => financialData.value.totalRevenue)
const financialOrders = computed(() => financialData.value.orderCount)
const financialProfit = computed(() => financialData.value.netProfit)
const financialMargin = computed(() => financialRevenue.value ? financialData.value.grossProfit / financialRevenue.value : 0)
const inventoryTotalProducts = computed(() => inventoryOverview.value.totalProducts)
const inventoryTotalQty = computed(() => inventoryOverview.value.totalQuantity)
const inventoryTotalValue = computed(() => inventoryOverview.value.totalValue)
const inventoryLowStock = computed(() => inventoryOverview.value.lowStockCount)
const productionYield = computed(() => 0.97)
const productionOutput = computed(() => 1280)
const productionDefects = computed(() => 38)

const formatMoney = (v: number | undefined | null) => {
  if (v == null) return '0.00'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(v)
}

const formatPercent = (v: number | undefined | null) => {
  if (v == null) return '0.00%'
  return (v * 100).toFixed(2) + '%'
}

async function loadSalesData() {
  try {
    const trendRes = await getSalesTrend(dateRange.value[0], dateRange.value[1])
    const trend = trendRes as any
    if (trend.dates && trend.amounts) {
      salesTrendRows.value = trend.dates.map((d: string, i: number) => ({
        date: d,
        amount: trend.amounts[i] || 0,
        orderCount: trend.orderCounts[i] || 0,
      }))
    }
  } catch (e) { console.error('Failed to load sales trend', e) }

  try {
    const catRes = await getCategoryDistribution(dateRange.value[0], dateRange.value[1])
    categoryList.value = (catRes as any)?.data || (Array.isArray(catRes) ? catRes : [])
  } catch (e) { console.error('Failed to load category distribution', e) }

  try {
    const regionRes = await getRegionalComparison(dateRange.value[0], dateRange.value[1])
    const r = regionRes as any
    if (r.rows) {
      regionList.value = r.rows.map((row: any) => ({
        region: row.region || row.dimension || '-',
        salesAmount: row.total_amount_sum || row.totalAmount || 0,
        orderCount: row.count || row.orderCount || 0,
      }))
    }
  } catch (e) { console.error('Failed to load regional comparison', e) }
}

async function loadFinancialData() {
  try {
    const res = await getFinancialSummary(dateRange.value[0], dateRange.value[1])
    const d = res as any
    financialData.value = {
      totalRevenue: d.todaySales || d.totalRevenue || 0,
      totalCost: (d.todaySales || 0) * 0.6,
      grossProfit: d.todayProfit || d.grossProfit || 0,
      netProfit: d.todayProfit || d.netProfit || 0,
      orderCount: d.todayOrders || d.orderCount || 0,
    }
  } catch (e) { console.error('Failed to load financial data', e) }
}

async function loadInventoryData() {
  try {
    const res = await getSafetyStockAlerts()
    safetyAlerts.value = (res as any)?.data || (Array.isArray(res) ? res : [])
  } catch (e) { console.error('Failed to load inventory data', e) }
}

async function loadAll() {
  loading.value = true
  await Promise.all([loadSalesData(), loadFinancialData(), loadInventoryData()])
  loading.value = false
}

onMounted(loadAll)
</script>

<style scoped>
.bi-dashboard { padding: 20px; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.page-header h2 { margin: 0; }
.stat-cards { margin-bottom: 20px; }
.stat-card { text-align: center; padding: 10px 0; }
.stat-value { font-size: 28px; font-weight: 700; color: #409EFF; }
.stat-value.warning { color: #E6A23C; }
.stat-label { font-size: 14px; color: #909399; margin-top: 8px; }
.section-card { margin-bottom: 20px; }
.section-row { margin-top: 20px; }
</style>
