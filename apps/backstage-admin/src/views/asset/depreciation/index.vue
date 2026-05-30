<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  generateDepreciation,
  listDepreciationByPeriod,
  getDepreciationStatistics,
  listAssets,
  type FixedAssetDepreciation,
  type DepreciationStatistics
} from '@/apis/asset'
import { MESSAGE_DURATION_SHORT } from '@/constants'
import { t } from '@/locales'

const loading = ref(false)
const statsLoading = ref(false)
const generateDialogVisible = ref(false)

// Statistics
const statistics = ref<DepreciationStatistics>({
  totalDepreciationAmount: 0,
  totalAccumulatedDepreciation: 0,
  periodDistribution: []
})

// List data
const depreciationList = ref<FixedAssetDepreciation[]>([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(10)

// Search params
const searchParams = reactive({
  period: ''
})

// Generate form
const generateForm = reactive({
  depreciationMethod: 'STRAIGHT_LINE',
  depreciationPeriod: ''
})

const depreciationMethodOptions = [
  { value: 'STRAIGHT_LINE', label: 'Straight Line' },
  { value: 'DOUBLE_DECLINING', label: 'Double Declining Balance' },
  { value: 'SUM_OF_YEARS', label: 'Sum of Years Digits' }
]

const formatMoney = (value: number | undefined | null) => {
  if (value === undefined || value === null) return '0.00'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
}

const formatDate = (date: string) => {
  if (!date) return '-'
  return date.split('T')[0]
}

const getMethodLabel = (method: string) => {
  const map: Record<string, string> = {
    STRAIGHT_LINE: 'Straight Line',
    DOUBLE_DECLINING: 'Double Declining',
    SUM_OF_YEARS: 'Sum of Years'
  }
  return map[method] || method
}

const getStatusLabel = (status: string) => {
  const map: Record<string, string> = {
    PENDING: 'Pending',
    POSTED: 'Posted',
    CANCELLED: 'Cancelled'
  }
  return map[status] || status
}

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING: 'warning',
    POSTED: 'success',
    CANCELLED: 'info'
  }
  return map[status] || 'info'
}

const loadStatistics = async () => {
  statsLoading.value = true
  try {
    const period = searchParams.period || new Date().toISOString().slice(0, 7)
    const response = await getDepreciationStatistics(period)
    statistics.value = response.data || statistics.value
  } catch (error) {
    console.error('Failed to load statistics:', error)
  } finally {
    statsLoading.value = false
  }
}

const loadData = async () => {
  loading.value = true
  try {
    const period = searchParams.period || new Date().toISOString().slice(0, 7)
    const response = await listDepreciationByPeriod(period)
    depreciationList.value = response.data || []
    total.value = (response.data || []).length
  } catch (error) {
    console.error('Failed to load depreciation list:', error)
    ElMessage.error('Failed to load depreciation list')
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  currentPage.value = 1
  loadData()
  loadStatistics()
}

const handleReset = () => {
  searchParams.period = ''
  currentPage.value = 1
  loadData()
  loadStatistics()
}

const handlePageChange = (page: number) => {
  currentPage.value = page
  loadData()
}

const handleSizeChange = (size: number) => {
  pageSize.value = size
  currentPage.value = 1
  loadData()
}

const handleGenerate = () => {
  generateForm.depreciationPeriod = new Date().toISOString().slice(0, 7)
  generateDialogVisible.value = true
}

const submitGenerate = async () => {
  try {
    await generateDepreciation(generateForm)
    ElMessage.success('Depreciation generated successfully')
    generateDialogVisible.value = false
    loadData()
    loadStatistics()
  } catch (error) {
    console.error('Failed to generate depreciation:', error)
    ElMessage.error('Failed to generate depreciation')
  }
}

onMounted(() => {
  searchParams.period = new Date().toISOString().slice(0, 7)
  loadData()
  loadStatistics()
})
</script>

<template>
  <div class="depreciation-container">
    <!-- Statistics Cards -->
    <el-row :gutter="20" class="stats-row">
      <el-col :span="12">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Total Depreciation Amount</div>
            <div class="stat-value">¥{{ formatMoney(statistics.totalDepreciationAmount) }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Total Accumulated Depreciation</div>
            <div class="stat-value">¥{{ formatMoney(statistics.totalAccumulatedDepreciation) }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Search Bar -->
    <el-card class="search-card">
      <el-form :model="searchParams" inline>
        <el-form-item label="Period">
          <el-date-picker
            v-model="searchParams.period"
            type="month"
            placeholder="Select period"
            value-format="YYYY-MM"
            @change="handleSearch"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Action Bar -->
    <el-card class="table-card">
      <div class="table-header">
        <el-button type="primary" @click="handleGenerate">Generate Depreciation</el-button>
      </div>

      <!-- Table -->
      <el-table :data="depreciationList" v-loading="loading" stripe>
        <el-table-column prop="assetCode" label="Asset Code" width="120" />
        <el-table-column prop="assetName" label="Asset Name" min-width="150" />
        <el-table-column prop="depreciationPeriod" label="Period" width="100" />
        <el-table-column prop="depreciationMethod" label="Method" width="150">
          <template #default="{ row }">
            {{ getMethodLabel(row.depreciationMethod) }}
          </template>
        </el-table-column>
        <el-table-column prop="originalValue" label="Original Value" width="130" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.originalValue) }}
          </template>
        </el-table-column>
        <el-table-column prop="depreciationAmount" label="Depreciation" width="130" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.depreciationAmount) }}
          </template>
        </el-table-column>
        <el-table-column prop="accumulatedDepreciation" label="Accumulated" width="130" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.accumulatedDepreciation) }}
          </template>
        </el-table-column>
        <el-table-column prop="netBookValue" label="Net Book Value" width="130" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.netBookValue) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="depreciationDate" label="Depreciation Date" width="120">
          <template #default="{ row }">
            {{ formatDate(row.depreciationDate) }}
          </template>
        </el-table-column>
      </el-table>

      <!-- Pagination -->
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @current-change="handlePageChange"
        @size-change="handleSizeChange"
        class="pagination"
      />
    </el-card>

    <!-- Generate Dialog -->
    <el-dialog v-model="generateDialogVisible" title="Generate Depreciation" width="500px" destroy-on-close>
      <el-form :model="generateForm" label-width="140px">
        <el-form-item label="Depreciation Method">
          <el-select v-model="generateForm.depreciationMethod" placeholder="Select method" style="width: 100%">
            <el-option v-for="item in depreciationMethodOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Depreciation Period">
          <el-date-picker
            v-model="generateForm.depreciationPeriod"
            type="month"
            placeholder="Select period"
            value-format="YYYY-MM"
            style="width: 100%"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="generateDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="submitGenerate">Generate</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.depreciation-container {
  padding: 20px;
}

.stats-row {
  margin-bottom: 20px;
}

.stat-card {
  text-align: center;
}

.stat-content {
  padding: 10px;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}

.search-card {
  margin-bottom: 20px;
}

.table-card {
  margin-bottom: 20px;
}

.table-header {
  margin-bottom: 15px;
}

.pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>