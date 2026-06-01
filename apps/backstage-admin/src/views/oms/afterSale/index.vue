<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Refresh, View, Check, Close, Wallet } from '@element-plus/icons-vue'
import {
  getAfterSaleListAPI,
  getAfterSaleByIdAPI,
  getAfterSaleStatisticsAPI,
  processAfterSaleAPI,
  batchApproveAfterSaleAPI,
  batchRejectAfterSaleAPI,
  type AfterSale,
  type AfterSaleQueryParam,
  type AfterSaleStatistic,
  type AfterSaleProcessParam
} from '@/apis/afterSale'
import { formatDateTime } from '@/utils/datetime'

const listQuery = ref<AfterSaleQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<AfterSale[]>([])
const listLoading = ref(true)
const total = ref(0)
const statistics = ref<AfterSaleStatistic | null>(null)

// Status options matching AfterSaleStatus enum
const statusOptions = [
  { label: 'Pending', value: 1, type: 'warning' },
  { label: 'Processing', value: 2, type: 'primary' },
  { label: 'Approved', value: 3, type: 'success' },
  { label: 'Rejected', value: 4, type: 'danger' },
  { label: 'Completed', value: 5, type: 'info' },
  { label: 'Cancelled', value: 6, type: 'info' }
]

// Type options matching AfterSaleType enum
const typeOptions = [
  { label: 'Return & Refund', value: 1 },
  { label: 'Exchange', value: 2 },
  { label: 'Refund Only', value: 3 }
]

const multipleSelection = ref<AfterSale[]>([])

// Dialog state
const dialogVisible = ref(false)
const dialogLoading = ref(false)
const detailMode = ref(false)
const currentAfterSale = ref<AfterSale | null>(null)

const processForm = ref<AfterSaleProcessParam>({
  id: 0,
  status: 3,
  rejectReason: '',
  remark: ''
})

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getAfterSaleListAPI(listQuery.value)
    listLoading.value = false
    const data = response.data as any
    list.value = data?.data || []
    total.value = data?.total || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error('Failed to load after-sale list')
  }
}

const getStatistics = async () => {
  try {
    const response = await getAfterSaleStatisticsAPI()
    statistics.value = response.data
  } catch (error) {
    ElMessage.error('Failed to load statistics')
  }
}

onMounted(() => {
  getList()
  getStatistics()
})

const handleSearch = () => {
  listQuery.value.pageNum = 1
  getList()
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10
  }
  getList()
}

const handleSelectionChange = (val: AfterSale[]) => {
  multipleSelection.value = val
}

const handleView = async (row: AfterSale) => {
  if (!row.id) return
  try {
    const response = await getAfterSaleByIdAPI(row.id)
    currentAfterSale.value = response.data
    detailMode.value = true
    dialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load after-sale details')
  }
}

const handleProcess = async (row: AfterSale) => {
  if (!row.id) return
  try {
    const response = await getAfterSaleByIdAPI(row.id)
    currentAfterSale.value = response.data
    processForm.value = {
      id: row.id,
      status: 3,
      rejectReason: '',
      remark: ''
    }
    detailMode.value = false
    dialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load after-sale details')
  }
}

const handleSubmitProcess = async () => {
  if (!currentAfterSale.value) return
  dialogLoading.value = true
  try {
    await processAfterSaleAPI(processForm.value)
    ElMessage.success('After-sale processed successfully')
    dialogVisible.value = false
    getList()
    getStatistics()
  } catch (error) {
    ElMessage.error('Failed to process after-sale')
  } finally {
    dialogLoading.value = false
  }
}

const handleBatchApprove = async () => {
  if (multipleSelection.value.length === 0) {
    ElMessage.warning('Please select items to approve')
    return
  }
  try {
    await ElMessageBox.confirm('Approve selected after-sale applications?', 'Confirm', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    const ids = multipleSelection.value.map(item => item.id!).filter(Boolean)
    await batchApproveAfterSaleAPI(ids)
    ElMessage.success('Batch approved successfully')
    getList()
    getStatistics()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to batch approve')
    }
  }
}

const handleBatchReject = async () => {
  if (multipleSelection.value.length === 0) {
    ElMessage.warning('Please select items to reject')
    return
  }
  try {
    const { value: reason } = await ElMessageBox.prompt('Please enter reject reason:', 'Batch Reject', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      inputPattern: /.+/,
      inputErrorMessage: 'Reason is required'
    })
    const ids = multipleSelection.value.map(item => item.id!).filter(Boolean)
    await batchRejectAfterSaleAPI(ids, reason)
    ElMessage.success('Batch rejected successfully')
    getList()
    getStatistics()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to batch reject')
    }
  }
}

const getStatusType = (status: number): 'success' | 'warning' | 'info' | 'danger' | 'primary' | undefined => {
  const map: Record<number, 'success' | 'warning' | 'info' | 'danger' | 'primary'> = {
    1: 'warning',
    2: 'primary',
    3: 'success',
    4: 'danger',
    5: 'info',
    6: 'info'
  }
  return map[status]
}

const formatStatus = (status: number): string => {
  return statusOptions.find(item => item.value === status)?.label || 'Unknown'
}

const formatType = (type: number): string => {
  return typeOptions.find(item => item.value === type)?.label || 'Unknown'
}

const formatAmount = (amount: number | undefined): string => {
  if (amount === undefined || amount === null) return '¥0.00'
  return `¥${(amount / 100).toFixed(2)}`
}
</script>

<template>
  <div class="after-sale-container">
    <!-- Statistics Cards -->
    <el-row :gutter="20" class="statistics-row">
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-value">{{ statistics?.totalCount || 0 }}</div>
            <div class="stat-label">Total Applications</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card stat-pending">
          <div class="stat-content">
            <div class="stat-value">{{ statistics?.pendingCount || 0 }}</div>
            <div class="stat-label">Pending</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card stat-processing">
          <div class="stat-content">
            <div class="stat-value">{{ statistics?.processingCount || 0 }}</div>
            <div class="stat-label">Processing</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card stat-refund">
          <div class="stat-content">
            <div class="stat-value">{{ formatAmount(statistics?.totalRefundAmount) }}</div>
            <div class="stat-label">Total Refund</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Search Card -->
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="Select status" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Type">
          <el-select v-model="listQuery.type" placeholder="Select type" clearable>
            <el-option v-for="item in typeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Order No">
          <el-input v-model="listQuery.orderNo" placeholder="Order No" clearable />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button :icon="Refresh" @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Table Card -->
    <el-card class="table-card">
      <div class="toolbar">
        <el-button type="success" :icon="Check" @click="handleBatchApprove" :disabled="multipleSelection.length === 0">
          Batch Approve
        </el-button>
        <el-button type="danger" :icon="Close" @click="handleBatchReject" :disabled="multipleSelection.length === 0">
          Batch Reject
        </el-button>
      </div>

      <el-table v-loading="listLoading" :data="list" border stripe @selection-change="handleSelectionChange">
        <el-table-column type="selection" width="55" />
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="orderNo" label="Order No" width="150" />
        <el-table-column prop="productName" label="Product" min-width="150" />
        <el-table-column prop="quantity" label="Qty" width="60" />
        <el-table-column label="Refund Amount" width="120">
          <template #default="{ row }">
            {{ formatAmount(row.refundAmount) }}
          </template>
        </el-table-column>
        <el-table-column prop="afterSaleTypeDesc" label="Type" width="120">
          <template #default="{ row }">
            {{ row.afterSaleTypeDesc || formatType(row.afterSaleType) }}
          </template>
        </el-table-column>
        <el-table-column prop="afterSaleStatusDesc" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.afterSaleStatus)">
              {{ row.afterSaleStatusDesc || formatStatus(row.afterSaleStatus) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="applyTime" label="Apply Time" width="170">
          <template #default="{ row }">
            {{ row.applyTime ? formatDateTime(row.applyTime) : '-' }}
          </template>
        </el-table-column>
        <el-table-column :label="t('common.actions')" width="180" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" :icon="View" @click="handleView(row)">View</el-button>
            <el-button v-if="row.afterSaleStatus === 1 || row.afterSaleStatus === 2" 
              link type="success" size="small" :icon="Check" @click="handleProcess(row)">Process</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="listQuery.pageNum"
        v-model:page-size="listQuery.pageSize"
        :total="total"
        :page-sizes="[10, 20, 50]"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="getList"
        @current-change="getList"
        style="margin-top: 20px; justify-content: flex-end"
      />
    </el-card>

    <!-- Detail/Process Dialog -->
    <el-dialog
      v-model="dialogVisible"
      :title="detailMode ? 'After-Sale Detail' : 'Process After-Sale'"
      width="700px"
      :close-on-click-modal="false"
    >
      <div v-if="currentAfterSale" v-loading="dialogLoading">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="ID">{{ currentAfterSale.id }}</el-descriptions-item>
          <el-descriptions-item label="Order No">{{ currentAfterSale.orderNo }}</el-descriptions-item>
          <el-descriptions-item label="Product">{{ currentAfterSale.productName }}</el-descriptions-item>
          <el-descriptions-item label="Quantity">{{ currentAfterSale.quantity }}</el-descriptions-item>
          <el-descriptions-item label="Refund Amount">{{ formatAmount(currentAfterSale.refundAmount) }}</el-descriptions-item>
          <el-descriptions-item label="Type">
            {{ currentAfterSale.afterSaleTypeDesc || formatType(currentAfterSale.afterSaleType) }}
          </el-descriptions-item>
          <el-descriptions-item label="Status">
            <el-tag :type="getStatusType(currentAfterSale.afterSaleStatus)">
              {{ currentAfterSale.afterSaleStatusDesc || formatStatus(currentAfterSale.afterSaleStatus) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="Apply Time">
            {{ currentAfterSale.applyTime ? formatDateTime(currentAfterSale.applyTime) : '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="Apply Reason" :span="2">
            {{ currentAfterSale.applyReason || '-' }}
          </el-descriptions-item>
          <el-descriptions-item v-if="currentAfterSale.rejectReason" label="Reject Reason" :span="2">
            {{ currentAfterSale.rejectReason }}
          </el-descriptions-item>
          <el-descriptions-item v-if="currentAfterSale.processTime" label="Process Time" :span="2">
            {{ formatDateTime(currentAfterSale.processTime) }}
          </el-descriptions-item>
          <el-descriptions-item v-if="currentAfterSale.remark" label="Remark" :span="2">
            {{ currentAfterSale.remark }}
          </el-descriptions-item>
        </el-descriptions>

        <!-- Process Form -->
        <el-form v-if="!detailMode" :model="processForm" label-width="120px" style="margin-top: 20px">
          <el-form-item label="Process Action">
            <el-radio-group v-model="processForm.status">
              <el-radio :value="3">Approve</el-radio>
              <el-radio :value="4">Reject</el-radio>
              <el-radio :value="5">Complete</el-radio>
            </el-radio-group>
          </el-form-item>
          <el-form-item v-if="processForm.status === 4" label="Reject Reason">
            <el-input v-model="processForm.rejectReason" type="textarea" :rows="3" placeholder="Enter reject reason" />
          </el-form-item>
          <el-form-item label="Remark">
            <el-input v-model="processForm.remark" type="textarea" :rows="3" placeholder="Enter remark" />
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <el-button @click="dialogVisible = false">Close</el-button>
        <el-button v-if="!detailMode" type="primary" :loading="dialogLoading" @click="handleSubmitProcess">
          Submit
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.after-sale-container {
  padding: 20px;
}

.statistics-row {
  margin-bottom: 20px;
}

.stat-card {
  text-align: center;
}

.stat-card :deep(.el-card__body) {
  padding: 20px;
}

.stat-content .stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
}

.stat-content .stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 8px;
}

.stat-pending :deep(.el-card__body) {
  background-color: #fdf6ec;
}

.stat-processing :deep(.el-card__body) {
  background-color: #ecf5ff;
}

.stat-refund :deep(.el-card__body) {
  background-color: #fef0f0;
}

.search-card {
  margin-bottom: 20px;
}

.table-card .toolbar {
  margin-bottom: 15px;
}

.table-card .toolbar .el-button {
  margin-right: 10px;
}
</style>