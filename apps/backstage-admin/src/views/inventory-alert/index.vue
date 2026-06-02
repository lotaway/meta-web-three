<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Refresh } from '@element-plus/icons-vue'
import {
  getInventoryAlertListAPI,
  getInventoryAlertStatisticsAPI,
  resolveInventoryAlertAPI,
  ignoreInventoryAlertAPI,
  type InventoryAlert,
  type InventoryAlertQueryParam,
  type InventoryAlertStatistics
} from '@/apis/inventoryAlert'
import { DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'

const loading = ref(false)
const statistics = ref<InventoryAlertStatistics>({
  total: 0,
  pending: 0,
  highPriority: 0,
  resolved: 0,
  ignored: 0
})

const query = reactive<InventoryAlertQueryParam>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  productId: undefined,
  productName: '',
  alertStatus: undefined,
  alertLevel: undefined,
  startDate: '',
  endDate: ''
})

const alertList = ref<InventoryAlert[]>([])
const total = ref(0)

const alertStatusOptions = [
  { label: 'Pending', value: 1 },
  { label: 'Notified', value: 2 },
  { label: 'Processing', value: 3 },
  { label: 'Resolved', value: 4 },
  { label: 'Ignored', value: 5 }
]

const alertLevelOptions = [
  { label: 'Low', value: 1 },
  { label: 'Medium', value: 2 },
  { label: 'High', value: 3 },
  { label: 'Critical', value: 4 }
]

const getAlertLevelTagType = (level: number): 'info' | 'warning' | 'danger' | 'success' => {
  const map: Record<number, 'info' | 'warning' | 'danger' | 'success'> = {
    1: 'info',
    2: 'warning',
    3: 'danger',
    4: 'success'
  }
  return map[level] || 'info'
}

const getAlertStatusTagType = (status: number): 'info' | 'warning' | 'danger' | 'success' | 'primary' => {
  const map: Record<number, 'info' | 'warning' | 'danger' | 'success' | 'primary'> = {
    1: 'warning',
    2: 'primary',
    3: 'info',
    4: 'success',
    5: 'info'
  }
  return map[status] || 'info'
}

const getAlertLevelText = (level: number): string => {
  const map: Record<number, string> = {
    1: 'Low',
    2: 'Medium',
    3: 'High',
    4: 'Critical'
  }
  return map[level] || 'Unknown'
}

const getAlertStatusText = (status: number): string => {
  const map: Record<number, string> = {
    1: 'Pending',
    2: 'Notified',
    3: 'Processing',
    4: 'Resolved',
    5: 'Ignored'
  }
  return map[status] || 'Unknown'
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getInventoryAlertListAPI(query)
    if (res.data) {
      alertList.value = res.data.list || []
      total.value = res.data.total || 0
    }
  } catch (error) {
    console.error('Failed to load alert list:', error)
  } finally {
    loading.value = false
  }
}

const getStatistics = async () => {
  try {
    const res = await getInventoryAlertStatisticsAPI()
    if (res.data) {
      statistics.value = res.data
    }
  } catch (error) {
    console.error('Failed to load statistics:', error)
  }
}

const handleSearch = () => {
  query.pageNum = 1
  getList()
}

const handleReset = () => {
  query.pageNum = 1
  query.productId = undefined
  query.productName = ''
  query.alertStatus = undefined
  query.alertLevel = undefined
  query.startDate = ''
  query.endDate = ''
  getList()
}

const handlePageChange = (page: number) => {
  query.pageNum = page
  getList()
}

const handleSizeChange = (size: number) => {
  query.pageSize = size
  query.pageNum = 1
  getList()
}

const handleRefresh = () => {
  getList()
  getStatistics()
}

const handleResolve = async (row: InventoryAlert) => {
  try {
    await ElMessageBox.confirm(
      `Are you sure you want to resolve this alert?`,
      'Confirm Resolve',
      {
        confirmButtonText: 'Confirm',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }
    )
    await resolveInventoryAlertAPI(row.id)
    ElMessage.success('Alert resolved successfully')
    getList()
    getStatistics()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to resolve alert:', error)
    }
  }
}

const handleIgnore = async (row: InventoryAlert) => {
  try {
    await ElMessageBox.confirm(
      `Are you sure you want to ignore this alert?`,
      'Confirm Ignore',
      {
        confirmButtonText: 'Confirm',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }
    )
    await ignoreInventoryAlertAPI(row.id)
    ElMessage.success('Alert ignored successfully')
    getList()
    getStatistics()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to ignore alert:', error)
    }
  }
}

onMounted(() => {
  getList()
  getStatistics()
})
</script>

<template>
  <div class="inventory-alert-container">
    <el-card class="statistics-card">
      <el-row :gutter="20">
        <el-col :span="4">
          <div class="stat-item">
            <div class="stat-value">{{ statistics.total }}</div>
            <div class="stat-label">Total Alerts</div>
          </div>
        </el-col>
        <el-col :span="4">
          <div class="stat-item stat-pending">
            <div class="stat-value">{{ statistics.pending }}</div>
            <div class="stat-label">Pending</div>
          </div>
        </el-col>
        <el-col :span="4">
          <div class="stat-item stat-high">
            <div class="stat-value">{{ statistics.highPriority }}</div>
            <div class="stat-label">High Priority</div>
          </div>
        </el-col>
        <el-col :span="4">
          <div class="stat-item stat-resolved">
            <div class="stat-value">{{ statistics.resolved }}</div>
            <div class="stat-label">Resolved</div>
          </div>
        </el-col>
        <el-col :span="4">
          <div class="stat-item stat-ignored">
            <div class="stat-value">{{ statistics.ignored }}</div>
            <div class="stat-label">Ignored</div>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <el-card class="filter-card">
      <el-form :inline="true" :model="query" class="filter-form">
        <el-form-item label="Product Name">
          <el-input v-model="query.productName" placeholder="Product Name" clearable style="width: 160px" />
        </el-form-item>
        <el-form-item label="Alert Status">
          <el-select v-model="query.alertStatus" placeholder="Select Status" clearable style="width: 140px">
            <el-option v-for="item in alertStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Alert Level">
          <el-select v-model="query.alertLevel" placeholder="Select Level" clearable style="width: 140px">
            <el-option v-for="item in alertLevelOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button :icon="Refresh" @click="handleReset">Reset</el-button>
          <el-button :icon="Refresh" @click="handleRefresh">Refresh</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="loading" :data="alertList" border style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="productName" label="Product Name" min-width="150" />
        <el-table-column prop="currentStock" label="Current Stock" width="120">
          <template #default="{ row }">
            <span :class="{ 'stock-warning': row.currentStock < row.threshold }">{{ row.currentStock }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="threshold" label="Threshold" width="100" />
        <el-table-column prop="alertLevel" label="Level" width="100">
          <template #default="{ row }">
            <el-tag :type="getAlertLevelTagType(row.alertLevel)">{{ getAlertLevelText(row.alertLevel) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="alertStatus" label="Status" width="110">
          <template #default="{ row }">
            <el-tag :type="getAlertStatusTagType(row.alertStatus)">{{ getAlertStatusText(row.alertStatus) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="alertMessage" label="Message" min-width="180" show-overflow-tooltip />
        <el-table-column prop="alertTime" label="Alert Time" width="170" />
        <el-table-column label="Actions" width="180" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="row.alertStatus !== 4 && row.alertStatus !== 5"
              type="success"
              link
              @click="handleResolve(row)"
            >
              Resolve
            </el-button>
            <el-button
              v-if="row.alertStatus !== 4 && row.alertStatus !== 5"
              type="info"
              link
              @click="handleIgnore(row)"
            >
              Ignore
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination-container">
        <el-pagination
          v-model:current-page="query.pageNum"
          v-model:page-size="query.pageSize"
          :page-sizes="PAGE_SIZE_OPTIONS"
          :total="total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.inventory-alert-container {
  padding: 20px;
}

.statistics-card {
  margin-bottom: 20px;
}

.stat-item {
  text-align: center;
  padding: 10px;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 5px;
}

.stat-pending .stat-value {
  color: #e6a23c;
}

.stat-high .stat-value {
  color: #f56c6c;
}

.stat-resolved .stat-value {
  color: #67c23a;
}

.stat-ignored .stat-value {
  color: #909399;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-form {
  margin-bottom: 0;
}

.table-card {
  min-height: 400px;
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.stock-warning {
  color: #f56c6c;
  font-weight: bold;
}
</style>