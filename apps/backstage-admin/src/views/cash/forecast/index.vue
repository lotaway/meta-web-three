<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listCashFlowForecastsAPI, deleteCashFlowForecastAPI } from '@/apis/cash'
import type { CashFlowForecast } from '@/apis/cash'
import { DEFAULT_PAGE_SIZE } from '@/constants'
import { t } from '@/locales'
import { Search, Plus, Delete } from '@element-plus/icons-vue'

const listQuery = reactive({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  forecastDate: ''
})
const list = ref<CashFlowForecast[]>([])
const total = ref(0)
const listLoading = ref(true)

const getList = async () => {
  listLoading.value = true
  try {
    const params: Record<string, any> = {}
    if (listQuery.forecastDate) params.forecastDate = listQuery.forecastDate
    
    const response = await listCashFlowForecastsAPI(params)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load forecast list:', error)
  }
}

const handleSearch = () => {
  listQuery.pageNum = 1
  getList()
}

const handleReset = () => {
  listQuery.forecastDate = ''
  listQuery.pageNum = 1
  getList()
}

const handleAdd = () => {
  ElMessage.info('Create page not implemented yet')
}

const handleDelete = async (row: CashFlowForecast) => {
  try {
    await ElMessageBox.confirm('Are you sure to delete this forecast?', 'Warning', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    await deleteCashFlowForecastAPI(row.id)
    ElMessage.success('Deleted successfully')
    getList()
  } catch (error) {
    console.error('Failed to delete forecast:', error)
  }
}

const formatMoney = (value: number | undefined | null) => {
  if (value === undefined || value === null) return '0.00'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
}

onMounted(() => {
  getList()
})
</script>

<template>
  <div class="forecast-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('cash.forecast.title') || 'Cash Flow Forecast Management' }}</span>
          <el-button type="primary" @click="handleAdd">
            <el-icon><Plus /></el-icon>
            {{ t('common.add') }}
          </el-button>
        </div>
      </template>
      
      <el-table v-loading="listLoading" :data="list" border style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="forecastNo" label="Forecast No." width="180" />
        <el-table-column prop="forecastDate" label="Forecast Date" width="150" />
        <el-table-column prop="startDate" label="Start Date" width="120" />
        <el-table-column prop="endDate" label="End Date" width="120" />
        <el-table-column prop="openingBalance" label="Opening Balance" width="150">
          <template #default="{ row }">
            ¥{{ formatMoney(row.openingBalance) }}
          </template>
        </el-table-column>
        <el-table-column prop="predictedInflow" label="Predicted Inflow" width="150">
          <template #default="{ row }">
            ¥{{ formatMoney(row.predictedInflow) }}
          </template>
        </el-table-column>
        <el-table-column prop="predictedOutflow" label="Predicted Outflow" width="150">
          <template #default="{ row }">
            ¥{{ formatMoney(row.predictedOutflow) }}
          </template>
        </el-table-column>
        <el-table-column prop="predictedClosingBalance" label="Predicted Closing" width="150">
          <template #default="{ row }">
            ¥{{ formatMoney(row.predictedClosingBalance) }}
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="Created At" width="180" />
        <el-table-column fixed="right" label="Operations" width="100">
          <template #default="{ row }">
            <el-button link type="danger" size="small" @click="handleDelete(row)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.forecast-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>