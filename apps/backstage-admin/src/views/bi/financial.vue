<template>
  <div class="bi-financial">
    <div class="page-header">
      <h2>{{ t('bi.financialAnalysis') }}</h2>
    </div>

    <el-row :gutter="20" class="stat-cards">
      <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatMoney(financialData.totalRevenue) }}</div><div class="stat-label">{{ t('bi.revenue') }}</div></div></el-card></el-col>
      <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ financialData.orderCount }}</div><div class="stat-label">{{ t('bi.totalOrders') }}</div></div></el-card></el-col>
      <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatMoney(financialData.netProfit) }}</div><div class="stat-label">{{ t('bi.netProfit') }}</div></div></el-card></el-col>
      <el-col :span="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatPercent(grossMargin) }}</div><div class="stat-label">{{ t('bi.grossMargin') }}</div></div></el-card></el-col>
    </el-row>

    <el-card v-loading="loading">
      <template #header><span>{{ t('bi.financialSummary') }}</span></template>
      <el-table :data="summaryRows" stripe empty-text="暂无数据">
        <el-table-column prop="metric" label="指标" width="120" />
        <el-table-column prop="value" label="数值"><template #default="{ row }">{{ row.value }}</template></el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { getFinancialSummary } from '@/apis/bi'
import type { FinancialSummary } from '@/apis/bi'

const loading = ref(false)
const financialData = ref<FinancialSummary>({ totalRevenue: 0, totalCost: 0, grossProfit: 0, netProfit: 0, orderCount: 0 })

const grossMargin = computed(() => financialData.value.totalRevenue ? financialData.value.grossProfit / financialData.value.totalRevenue : 0)

const summaryRows = computed(() => [
  { metric: t('bi.revenue'), value: formatMoney(financialData.value.totalRevenue) },
  { metric: t('bi.totalOrders'), value: financialData.value.orderCount.toString() },
  { metric: t('bi.netProfit'), value: formatMoney(financialData.value.netProfit) },
  { metric: t('bi.grossMargin'), value: formatPercent(grossMargin.value) },
])

const formatMoney = (v: number) => new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2 }).format(v || 0)
const formatPercent = (v: number) => (v * 100).toFixed(2) + '%'

async function loadData() {
  loading.value = true
  try {
    const res = await getFinancialSummary('', '')
    const d = res as any
    financialData.value = {
      totalRevenue: d.todaySales || d.totalRevenue || 0,
      totalCost: (d.todaySales || 0) * 0.6,
      grossProfit: d.todayProfit || d.grossProfit || 0,
      netProfit: d.todayProfit || d.netProfit || 0,
      orderCount: d.todayOrders || d.orderCount || 0,
    }
  } catch (e) { console.error(e); ElMessage.error('Failed to load financial data') }
  loading.value = false
}

onMounted(loadData)
</script>

<style scoped>
.bi-financial { padding: 20px; }
.page-header { margin-bottom: 20px; }
.page-header h2 { margin: 0; }
.stat-cards { margin-bottom: 20px; }
.stat-card { text-align: center; padding: 10px 0; }
.stat-value { font-size: 28px; font-weight: 700; color: #409EFF; }
.stat-label { font-size: 14px; color: #909399; margin-top: 8px; }
</style>
