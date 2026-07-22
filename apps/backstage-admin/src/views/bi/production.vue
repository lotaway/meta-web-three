<template>
  <div class="bi-production">
    <div class="page-header">
      <h2>{{ t('bi.productionAnalysis') }}</h2>
      <el-date-picker v-model="dateRange" type="daterange"
        :start-placeholder="t('bi.startDate')" :end-placeholder="t('bi.endDate')"
        value-format="YYYY-MM-DD" @change="loadData" />
    </div>

    <div v-loading="loading" class="stat-cards-wrapper">
    <el-row :gutter="20" class="stat-cards">
      <el-col :xs="24" :sm="12" :md="8"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatPercent(summaryYield) }}</div><div class="stat-label">{{ t('bi.yieldRate') }}</div></div></el-card></el-col>
      <el-col :xs="24" :sm="12" :md="8"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ totalOutput }}</div><div class="stat-label">{{ t('bi.productionOutput') }}</div></div></el-card></el-col>
      <el-col :xs="24" :sm="12" :md="8"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ totalDefects }}</div><div class="stat-label">{{ t('bi.defectCount') }}</div></div></el-card></el-col>
    </el-row>
    </div>

    <el-card class="section-card" v-loading="loading">
      <template #header><span>{{ t('bi.yieldRate') }}</span></template>
      <el-table :data="yieldRows" stripe size="small" empty-text="暂无数据">
        <el-table-column prop="period" :label="t('bi.dateRange')" />
        <el-table-column prop="yieldRate" :label="t('bi.yieldRate')" width="120"><template #default="{ row }">{{ formatPercent(row.yieldRate) }}</template></el-table-column>
        <el-table-column prop="totalProduced" :label="t('bi.productionOutput')" width="120" />
        <el-table-column prop="defective" :label="t('bi.defectCount')" width="100" />
      </el-table>
    </el-card>

    <el-card v-loading="loading" class="section-card">
      <template #header><span>{{ t('bi.oee') }}</span></template>
      <el-table :data="oeeRows" stripe size="small" empty-text="暂无数据">
        <el-table-column prop="period" :label="t('bi.dateRange')" />
        <el-table-column prop="availability" :label="t('bi.availability')" width="120"><template #default="{ row }">{{ formatPercent(row.availability) }}</template></el-table-column>
        <el-table-column prop="performance" :label="t('bi.performance')" width="120"><template #default="{ row }">{{ formatPercent(row.performance) }}</template></el-table-column>
        <el-table-column prop="quality" :label="t('bi.quality')" width="120"><template #default="{ row }">{{ formatPercent(row.quality) }}</template></el-table-column>
        <el-table-column prop="oee" :label="t('bi.oee')" width="120"><template #default="{ row }">{{ formatPercent(row.oee) }}</template></el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { getProductionAnalytics, getYieldRate, getOeeAnalysis } from '@/apis/bi'
import type { YieldRate, OeeAnalysis } from '@/apis/bi'

const defaultStart = new Date(new Date().getFullYear(), 0, 1).toISOString().split('T')[0]
const defaultEnd = new Date().toISOString().split('T')[0]
const dateRange = ref<string[]>([defaultStart, defaultEnd])
const loading = ref(false)
const yieldRows = ref<YieldRate[]>([])
const oeeRows = ref<OeeAnalysis[]>([])
const productionMeta = ref({ totalOutput: 0, defectCount: 0, yieldRate: 0 })

const summaryYield = computed(() => productionMeta.value.yieldRate)
const totalOutput = computed(() => productionMeta.value.totalOutput)
const totalDefects = computed(() => productionMeta.value.defectCount)

const formatPercent = (v: number) => (v * 100).toFixed(2) + '%'

async function loadProductionMeta() {
  try {
    const res = await getProductionAnalytics(dateRange.value[0], dateRange.value[1])
    const p = res.data
    productionMeta.value = {
      totalOutput: p.totalOutput || 0,
      defectCount: p.defectCount || 0,
      yieldRate: p.yieldRate ?? 0,
    }
  } catch (e) { console.error(e); ElMessage.error('Failed to load production data') }
}

async function loadYield() {
  try {
    const res = await getYieldRate()
    yieldRows.value = res.data || []
  } catch (e) { console.error(e); ElMessage.error('Failed to load yield rate') }
}

async function loadOee() {
  try {
    const res = await getOeeAnalysis()
    oeeRows.value = res.data || []
  } catch (e) { console.error(e); ElMessage.error('Failed to load OEE analysis') }
}

async function loadData() {
  loading.value = true
  await Promise.all([loadProductionMeta(), loadYield(), loadOee()])
  loading.value = false
}

onMounted(loadData)
</script>

<style scoped>
.bi-production { padding: 20px; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.page-header h2 { margin: 0; }
.stat-cards-wrapper { margin-bottom: 20px; min-height: 120px; }
.stat-card { text-align: center; padding: 10px 0; }
.stat-value { font-size: 28px; font-weight: 700; color: #409EFF; }
.stat-label { font-size: 14px; color: #909399; margin-top: 8px; }
.section-card { margin-bottom: 20px; }
</style>
