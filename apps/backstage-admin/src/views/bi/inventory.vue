<template>
  <div class="bi-inventory">
    <div class="page-header">
      <h2>{{ t('bi.inventoryAnalysis') }}</h2>
    </div>

    <div v-loading="loading" class="stat-cards-wrapper">
    <el-row :gutter="20" class="stat-cards">
      <el-col :xs="24" :sm="12" :md="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ totalProducts }}</div><div class="stat-label">{{ t('bi.totalProducts') }}</div></div></el-card></el-col>
      <el-col :xs="24" :sm="12" :md="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ totalQty }}</div><div class="stat-label">{{ t('bi.totalQuantity') }}</div></div></el-card></el-col>
      <el-col :xs="24" :sm="12" :md="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value">{{ formatMoney(inventoryValue) }}</div><div class="stat-label">{{ t('bi.inventoryValue') }}</div></div></el-card></el-col>
      <el-col :xs="24" :sm="12" :md="6"><el-card shadow="hover"><div class="stat-card"><div class="stat-value warning">{{ alertCount }}</div><div class="stat-label">{{ t('bi.safetyStockAlert') }}</div></div></el-card></el-col>
    </el-row>
    </div>

    <el-card class="section-card" v-loading="loading">
      <template #header><span>{{ t('bi.safetyStockAlert') }}</span></template>
      <el-table :data="alerts" stripe size="small" empty-text="暂无数据">
        <el-table-column prop="productName" :label="t('bi.productName')" />
        <el-table-column prop="warehouseName" :label="t('bi.warehouse')" />
        <el-table-column prop="quantity" :label="t('bi.currentStock')" width="100" />
        <el-table-column prop="minStock" :label="t('bi.safetyStock')" width="100" />
      </el-table>
    </el-card>

    <el-card v-loading="loading" class="section-card">
      <template #header><span>{{ t('bi.turnoverRate') }}</span></template>
      <el-table :data="turnoverRows" stripe size="small" empty-text="暂无数据">
        <el-table-column prop="period" :label="t('bi.dateRange')" />
        <el-table-column prop="turnoverRate" :label="t('bi.turnoverRate')" width="120"><template #default="{ row }">{{ row.turnoverRate.toFixed(2) }}</template></el-table-column>
        <el-table-column prop="daysInInventory" :label="t('bi.daysInInventory')" width="120"><template #default="{ row }">{{ row.daysInInventory.toFixed(1) }}</template></el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { getSafetyStockAlerts, getInventoryTurnover, getAbcAnalysis } from '@/apis/bi'
import type { SafetyStockAlert, InventoryTurnover } from '@/apis/bi'

const loading = ref(false)
const alerts = ref<SafetyStockAlert[]>([])
const turnoverRows = ref<InventoryTurnover[]>([])
const abcValue = ref(0)

const totalProducts = computed(() => alerts.value.length)
const totalQty = computed(() => alerts.value.reduce((s, r) => s + r.quantity, 0))
const inventoryValue = computed(() => abcValue.value)
const alertCount = computed(() => alerts.value.filter(r => r.quantity < r.minStock).length)

const formatMoney = (v: number) => new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2 }).format(v || 0)

async function loadAbc() {
  try {
    const res = await getAbcAnalysis() as { rows?: Array<{ inventoryValue?: number }> }
    const rows = res?.rows || []
    abcValue.value = rows.reduce((s, r) => s + (r.inventoryValue || 0), 0)
  } catch { abcValue.value = 0 }
}

async function loadAlerts() {
  try {
    const res = await getSafetyStockAlerts()
    alerts.value = res.data || []
  } catch (e) { console.error(e); ElMessage.error('Failed to load safety stock alerts') }
}

async function loadTurnover() {
  try {
    const res = await getInventoryTurnover()
    turnoverRows.value = res.data || []
  } catch (e) { console.error(e); ElMessage.error('Failed to load inventory turnover') }
}

async function loadData() {
  loading.value = true
  await Promise.all([loadAlerts(), loadTurnover(), loadAbc()])
  loading.value = false
}

onMounted(loadData)
</script>

<style scoped>
.bi-inventory { padding: 20px; }
.page-header { margin-bottom: 20px; }
.page-header h2 { margin: 0; }
.stat-cards-wrapper { margin-bottom: 20px; min-height: 120px; }
.stat-card { text-align: center; padding: 10px 0; }
.stat-value { font-size: 28px; font-weight: 700; color: #409EFF; }
.stat-value.warning { color: #E6A23C; }
.stat-label { font-size: 14px; color: #909399; margin-top: 8px; }
.section-card { margin-bottom: 20px; }
</style>
