<template>
  <div class="bi-sales">
    <div class="page-header">
      <h2>{{ t('bi.salesAnalysis') }}</h2>
      <el-date-picker v-model="dateRange" type="daterange"
        :start-placeholder="t('bi.startDate')" :end-placeholder="t('bi.endDate')"
        value-format="YYYY-MM-DD" @change="loadData" />
    </div>

    <el-row :gutter="20" class="stat-cards">
      <el-col :xs="24" :sm="12" :md="8"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatMoney(totalAmount) }}</div><div class="stat-label">{{ t('bi.totalRevenue') }}</div></div></el-card></el-col>
      <el-col :xs="24" :sm="12" :md="8"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ totalOrders }}</div><div class="stat-label">{{ t('bi.totalOrders') }}</div></div></el-card></el-col>
    </el-row>

    <el-card v-loading="loading">
      <template #header><span>{{ t('bi.salesTrend') }}</span></template>
      <el-table :data="trendRows" stripe empty-text="暂无数据">
        <el-table-column prop="date" label="日期" width="120" />
        <el-table-column prop="amount" label="销售额" width="150"><template #default="{ row }">{{ formatMoney(row.amount) }}</template></el-table-column>
        <el-table-column prop="orderCount" label="订单数" width="100" />
      </el-table>
    </el-card>

    <el-row :gutter="20" class="section-row">
      <el-col :xs="24" :sm="12">
        <el-card v-loading="loading">
          <template #header><span>{{ t('bi.categoryDistribution') }}</span></template>
          <el-table :data="categoryList" stripe size="small" max-height="300" empty-text="暂无数据">
            <el-table-column prop="categoryName" label="品类" />
            <el-table-column prop="salesAmount" label="销售额" width="120"><template #default="{ row }">{{ formatMoney(row.salesAmount) }}</template></el-table-column>
            <el-table-column prop="growthRate" label="增长率" width="80"><template #default="{ row }">{{ (row.growthRate * 100).toFixed(1) }}%</template></el-table-column>
          </el-table>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="12">
        <el-card v-loading="loading">
          <template #header><span>{{ t('bi.regionalComparison') }}</span></template>
          <el-table :data="regionRows" stripe size="small" max-height="300" empty-text="暂无数据">
            <el-table-column prop="region" :label="t('bi.region')" />
            <el-table-column prop="salesAmount" :label="t('bi.salesAmount')" width="120"><template #default="{ row }">{{ formatMoney(row.salesAmount) }}</template></el-table-column>
            <el-table-column prop="orderCount" :label="t('bi.orderCount')" width="80" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { getSalesTrend, getCategoryDistribution, getRegionalComparison } from '@/apis/bi'
import type { CategoryDistribution } from '@/apis/bi'

const defaultStart = new Date(new Date().getFullYear(), 0, 1).toISOString().split('T')[0]
const defaultEnd = new Date().toISOString().split('T')[0]
const dateRange = ref<string[]>([defaultStart, defaultEnd])
const loading = ref(false)
const trendRows = ref<{ date: string; amount: number; orderCount: number }[]>([])
const categoryList = ref<CategoryDistribution[]>([])
const regionRows = ref<{ region: string; salesAmount: number; orderCount: number }[]>([])

const totalAmount = computed(() => trendRows.value.reduce((s, r) => s + r.amount, 0))
const totalOrders = computed(() => trendRows.value.reduce((s, r) => s + r.orderCount, 0))

const formatMoney = (v: number) => new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2 }).format(v || 0)

async function loadSalesTrend() {
  try {
    const trendRes = await getSalesTrend(dateRange.value[0], dateRange.value[1])
    const trend = trendRes as any
    if (trend.dates) {
      trendRows.value = trend.dates.map((d: string, i: number) => ({
        date: d, amount: trend.amounts[i] || 0, orderCount: trend.orderCounts[i] || 0,
      }))
    }
  } catch (e) { console.error(e); ElMessage.error('Failed to load sales trend') }
}

async function loadCategoryData() {
  try {
    const catRes = await getCategoryDistribution(dateRange.value[0], dateRange.value[1])
    categoryList.value = (catRes as any)?.data || (Array.isArray(catRes) ? catRes : [])
  } catch (e) { console.error(e); ElMessage.error('Failed to load category distribution') }
}

async function loadRegionalData() {
  try {
    const regionRes = await getRegionalComparison(dateRange.value[0], dateRange.value[1])
    const r = regionRes as any
    if (r.rows) {
      regionRows.value = r.rows.map((row: any) => ({
        region: row.region || row.dimension || '-',
        salesAmount: row.total_amount_sum || 0,
        orderCount: row.count || 0,
      }))
    }
  } catch (e) { console.error(e); ElMessage.error('Failed to load regional comparison') }
}

async function loadData() {
  loading.value = true
  await Promise.all([loadSalesTrend(), loadCategoryData(), loadRegionalData()])
  loading.value = false
}

onMounted(loadData)
</script>

<style scoped>
.bi-sales { padding: 20px; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.page-header h2 { margin: 0; }
.stat-cards { margin-bottom: 20px; }
.stat-card { text-align: center; padding: 10px 0; }
.stat-value { font-size: 28px; font-weight: 700; color: #409EFF; }
.stat-label { font-size: 14px; color: #909399; margin-top: 8px; }
.section-row { margin-top: 20px; }
</style>
