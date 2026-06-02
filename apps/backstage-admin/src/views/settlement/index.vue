<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Refresh, Edit, Delete, Check, Close, Money, Warning, CircleCheck, CircleClose } from '@element-plus/icons-vue'
import {
  getSettlementListAPI,
  getSettlementDetailAPI,
  createSettlementAPI,
  confirmSettlementAPI,
  processSettlementAPI,
  completeSettlementAPI,
  failSettlementAPI,
  cancelSettlementAPI,
  refundSettlementAPI,
  type SettlementOrder,
  type SettlementQueryParam,
  type SettlementRequest,
  SettlementStatus,
  SettlementStatusText
} from '@/apis/settlement'
import { DEFAULT_PAGE_SIZE, MESSAGE_DURATION_SHORT } from '@/constants'
import { t } from '@/locales'
import { formatDateTime, formatNumber } from '@/utils/format'

const listQuery = reactive<SettlementQueryParam>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  status: undefined,
  merchantId: undefined,
  startDate: undefined,
  endDate: undefined
})

const list = ref<SettlementOrder[]>([])
const total = ref(0)
const listLoading = ref(false)

const activeTab = ref('list')
const detailLoading = ref(false)
const currentSettlement = ref<SettlementOrder | null>(null)

const dialogVisible = ref(false)
const dialogLoading = ref(false)
const dialogTitle = ref('')

const formData = reactive<SettlementRequest>({
  settlementNo: '',
  orderNo: '',
  merchantId: 0,
  merchantName: '',
  orderAmount: 0,
  commissionRate: 0.05
})

const statusOptions = [
  { value: '', label: 'common.all' },
  { value: SettlementStatus.PENDING, label: 'settlement.status.pending' },
  { value: SettlementStatus.CONFIRMED, label: 'settlement.status.confirmed' },
  { value: SettlementStatus.PROCESSING, label: 'settlement.status.processing' },
  { value: SettlementStatus.COMPLETED, label: 'settlement.status.completed' },
  { value: SettlementStatus.FAILED, label: 'settlement.status.failed' },
  { value: SettlementStatus.CANCELLED, label: 'settlement.status.cancelled' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const params: SettlementQueryParam = {
      pageNum: listQuery.pageNum,
      pageSize: listQuery.pageSize
    }
    if (listQuery.status) params.status = listQuery.status
    if (listQuery.merchantId) params.merchantId = listQuery.merchantId
    if (listQuery.startDate) params.startDate = listQuery.startDate
    if (listQuery.endDate) params.endDate = listQuery.endDate

    const response = await getSettlementListAPI(params)
    list.value = response.data.list || []
    total.value = response.data.total || 0
  } catch (error) {
    console.error('Failed to load settlement list:', error)
    ElMessage.error(t('common.queryFailed'))
  } finally {
    listLoading.value = false
  }
}

const handleSearch = () => {
  listQuery.pageNum = 1
  getList()
}

const handleReset = () => {
  listQuery.status = undefined
  listQuery.merchantId = undefined
  listQuery.startDate = undefined
  listQuery.endDate = undefined
  listQuery.pageNum = 1
  getList()
}

const handleView = async (row: SettlementOrder) => {
  detailLoading.value = true
  try {
    const response = await getSettlementDetailAPI(row.id)
    currentSettlement.value = response.data
    activeTab.value = 'detail'
  } catch (error) {
    console.error('Failed to load settlement detail:', error)
    ElMessage.error(t('common.queryFailed'))
  } finally {
    detailLoading.value = false
  }
}

const handleAdd = () => {
  formData.settlementNo = `ST${Date.now()}`
  formData.orderNo = ''
  formData.merchantId = 0
  formData.merchantName = ''
  formData.orderAmount = 0
  formData.commissionRate = 0.05
  dialogTitle.value = t('settlement.create')
  dialogVisible.value = true
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    await createSettlementAPI(formData)
    ElMessage.success(t('common.createSuccess'))
    dialogVisible.value = false
    getList()
  } catch (error) {
    console.error('Failed to create settlement:', error)
    ElMessage.error(t('common.createFailed'))
  } finally {
    dialogLoading.value = false
  }
}

const handleConfirm = async (row: SettlementOrder | null) => {
  if (!row) return
  try {
    await ElMessageBox.confirm(t('settlement.confirmMsg'), t('common.warning'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await confirmSettlementAPI(row.id)
    ElMessage.success(t('common.operationSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to confirm settlement:', error)
    }
  }
}

const handleProcess = async (row: SettlementOrder | null) => {
  if (!row) return
  try {
    await ElMessageBox.confirm(t('settlement.processMsg'), t('common.warning'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await processSettlementAPI(row.id)
    ElMessage.success(t('common.operationSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to process settlement:', error)
    }
  }
}

const handleComplete = async (row: SettlementOrder | null) => {
  if (!row) return
  try {
    await ElMessageBox.confirm(t('settlement.completeMsg'), t('common.warning'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await completeSettlementAPI(row.id)
    ElMessage.success(t('common.operationSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to complete settlement:', error)
    }
  }
}

const handleFail = async (row: SettlementOrder | null) => {
  if (!row) return
  try {
    const { value: reason } = await ElMessageBox.prompt(t('settlement.failReason'), t('settlement.fail'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      inputPattern: /.+/,
      inputErrorMessage: 'Reason is required'
    })
    await failSettlementAPI(row.id, reason)
    ElMessage.success(t('common.operationSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to fail settlement:', error)
    }
  }
}

const handleCancel = async (row: SettlementOrder | null) => {
  if (!row) return
  try {
    await ElMessageBox.confirm(t('settlement.cancelMsg'), t('common.warning'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await cancelSettlementAPI(row.id)
    ElMessage.success(t('common.operationSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to cancel settlement:', error)
    }
  }
}

const handleRefund = async (row: SettlementOrder | null) => {
  if (!row) return
  try {
    const { value: amount } = await ElMessageBox.prompt(t('settlement.refundAmount'), t('settlement.refund'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      inputPattern: /^\d+(\.\d{1,2})?$/,
      inputErrorMessage: 'Please enter a valid amount'
    })
    await refundSettlementAPI(row.id, parseFloat(amount))
    ElMessage.success(t('common.operationSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to refund settlement:', error)
    }
  }
}

const handleBack = () => {
  activeTab.value = 'list'
  currentSettlement.value = null
}

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const statusMap: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING: 'info',
    CONFIRMED: 'primary',
    PROCESSING: 'warning',
    COMPLETED: 'success',
    FAILED: 'danger',
    CANCELLED: 'info'
  }
  return statusMap[status] || 'info'
}

const getStatusText = (status: string) => {
  return t(SettlementStatusText[status] || status)
}

onMounted(() => {
  getList()
})
</script>

<template>
  <div class="settlement-container">
    <el-card v-if="activeTab === 'list'" class="settlement-card">
      <template #header>
        <div class="card-header">
          <span>{{ t('settlement.title') }}</span>
          <el-button type="primary" :icon="Plus" @click="handleAdd">
            {{ t('common.create') }}
          </el-button>
        </div>
      </template>

      <!-- Search Form -->
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item :label="t('settlement.status')">
          <el-select v-model="listQuery.status" :placeholder="t('common.select')" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="t(item.label)" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('settlement.merchantId')">
          <el-input v-model="listQuery.merchantId" :placeholder="t('settlement.merchantId')" clearable />
        </el-form-item>
        <el-form-item :label="t('settlement.dateRange')">
          <el-date-picker
            v-model="listQuery.startDate"
            type="datetime"
            :placeholder="t('common.startDate')"
            value-format="YYYY-MM-DD HH:mm:ss"
          />
          <span class="date-separator">-</span>
          <el-date-picker
            v-model="listQuery.endDate"
            type="datetime"
            :placeholder="t('common.endDate')"
            value-format="YYYY-MM-DD HH:mm:ss"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">{{ t('common.search') }}</el-button>
          <el-button :icon="Refresh" @click="handleReset">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>

      <!-- Table -->
      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="settlementNo" :label="t('settlement.settlementNo')" width="180" />
        <el-table-column prop="orderNo" :label="t('settlement.orderNo')" width="180" />
        <el-table-column prop="merchantName" :label="t('settlement.merchantName')" width="150" />
        <el-table-column prop="orderAmount" :label="t('settlement.orderAmount')" width="120">
          <template #default="{ row }">
            {{ formatNumber(row.orderAmount) }}
          </template>
        </el-table-column>
        <el-table-column prop="settlementAmount" :label="t('settlement.settlementAmount')" width="120">
          <template #default="{ row }">
            {{ formatNumber(row.settlementAmount) }}
          </template>
        </el-table-column>
        <el-table-column prop="commissionAmount" :label="t('settlement.commissionAmount')" width="120">
          <template #default="{ row }">
            {{ formatNumber(row.commissionAmount) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" :label="t('settlement.status')" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" :label="t('settlement.createdAt')" width="160">
          <template #default="{ row }">
            {{ formatDateTime(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('common.operate')" width="200" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">
              {{ t('common.view') }}
            </el-button>
            <el-button v-if="row.status === 'PENDING'" link type="primary" size="small" @click="handleConfirm(row)">
              {{ t('settlement.confirm') }}
            </el-button>
            <el-button v-if="row.status === 'CONFIRMED'" link type="warning" size="small" @click="handleProcess(row)">
              {{ t('settlement.process') }}
            </el-button>
            <el-button v-if="row.status === 'PROCESSING'" link type="success" size="small" @click="handleComplete(row)">
              {{ t('settlement.complete') }}
            </el-button>
            <el-button v-if="row.status === 'PROCESSING'" link type="danger" size="small" @click="handleFail(row)">
              {{ t('settlement.fail') }}
            </el-button>
            <el-button v-if="row.status === 'PENDING' || row.status === 'CONFIRMED'" link type="danger" size="small" @click="handleCancel(row)">
              {{ t('settlement.cancel') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- Pagination -->
      <div class="pagination-container">
        <el-pagination
          v-model:current-page="listQuery.pageNum"
          v-model:page-size="listQuery.pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="getList"
          @current-change="getList"
        />
      </div>
    </el-card>

    <!-- Detail Card -->
    <el-card v-if="activeTab === 'detail'" class="detail-card">
      <template #header>
        <div class="card-header">
          <span>{{ t('settlement.detail') }}</span>
          <el-button @click="handleBack">{{ t('common.back') }}</el-button>
        </div>
      </template>

      <el-descriptions v-loading="detailLoading" :column="2" border>
        <el-descriptions-item :label="t('settlement.id')">
          {{ currentSettlement?.id }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.settlementNo')">
          {{ currentSettlement?.settlementNo }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.orderNo')">
          {{ currentSettlement?.orderNo }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.merchantName')">
          {{ currentSettlement?.merchantName }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.orderAmount')">
          {{ formatNumber(currentSettlement?.orderAmount || 0) }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.settlementAmount')">
          {{ formatNumber(currentSettlement?.settlementAmount || 0) }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.commissionAmount')">
          {{ formatNumber(currentSettlement?.commissionAmount || 0) }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.refundAmount')">
          {{ formatNumber(currentSettlement?.refundAmount || 0) }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.status')">
          <el-tag :type="getStatusType(currentSettlement?.status || '')">
            {{ getStatusText(currentSettlement?.status || '') }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.channel')">
          {{ currentSettlement?.channel || '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.settlementDate')">
          {{ currentSettlement?.settlementDate ? formatDateTime(currentSettlement.settlementDate) : '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.createdAt')">
          {{ formatDateTime(currentSettlement?.createdAt) }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('settlement.updatedAt')" :span="2">
          {{ formatDateTime(currentSettlement?.updatedAt) }}
        </el-descriptions-item>
        <el-descriptions-item v-if="currentSettlement?.description" :label="t('settlement.description')" :span="2">
          {{ currentSettlement.description }}
        </el-descriptions-item>
      </el-descriptions>

      <!-- Actions -->
      <div class="detail-actions">
        <el-button v-if="currentSettlement?.status === 'PENDING'" type="primary" :icon="Check" @click="handleConfirm(currentSettlement)">
          {{ t('settlement.confirm') }}
        </el-button>
        <el-button v-if="currentSettlement?.status === 'CONFIRMED'" type="warning" :icon="Money" @click="handleProcess(currentSettlement)">
          {{ t('settlement.process') }}
        </el-button>
        <el-button v-if="currentSettlement?.status === 'PROCESSING'" type="success" :icon="CircleCheck" @click="handleComplete(currentSettlement)">
          {{ t('settlement.complete') }}
        </el-button>
        <el-button v-if="currentSettlement?.status === 'PROCESSING'" type="danger" :icon="CircleClose" @click="handleFail(currentSettlement)">
          {{ t('settlement.fail') }}
        </el-button>
        <el-button v-if="currentSettlement?.status === 'PENDING' || currentSettlement?.status === 'CONFIRMED'" type="danger" :icon="Close" @click="handleCancel(currentSettlement)">
          {{ t('settlement.cancel') }}
        </el-button>
        <el-button v-if="currentSettlement?.status !== 'COMPLETED' && currentSettlement?.status !== 'CANCELLED'" type="info" :icon="Money" @click="handleRefund(currentSettlement)">
          {{ t('settlement.refund') }}
        </el-button>
      </div>
    </el-card>

    <!-- Create Dialog -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="500px">
      <el-form :model="formData" label-width="120px">
        <el-form-item :label="t('settlement.settlementNo')">
          <el-input v-model="formData.settlementNo" disabled />
        </el-form-item>
        <el-form-item :label="t('settlement.orderNo')" required>
          <el-input v-model="formData.orderNo" :placeholder="t('settlement.orderNo')" />
        </el-form-item>
        <el-form-item :label="t('settlement.merchantId')" required>
          <el-input-number v-model="formData.merchantId" :min="1" />
        </el-form-item>
        <el-form-item :label="t('settlement.merchantName')" required>
          <el-input v-model="formData.merchantName" :placeholder="t('settlement.merchantName')" />
        </el-form-item>
        <el-form-item :label="t('settlement.orderAmount')" required>
          <el-input-number v-model="formData.orderAmount" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item :label="t('settlement.commissionRate')" required>
          <el-input-number v-model="formData.commissionRate" :min="0" :max="1" :step="0.01" :precision="2" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleSubmit">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.settlement-container {
  padding: 20px;
}

.settlement-card,
.detail-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.search-form {
  margin-bottom: 20px;
}

.date-separator {
  margin: 0 10px;
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.detail-actions {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}
</style>