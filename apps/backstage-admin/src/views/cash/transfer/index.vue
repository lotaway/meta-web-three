<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listCashTransfersAPI, deleteCashTransferAPI, submitCashTransferAPI, approveCashTransferAPI, rejectCashTransferAPI, cancelCashTransferAPI } from '@/apis/cash'
import type { CashTransfer } from '@/apis/cash'
import { DEFAULT_PAGE_SIZE } from '@/constants'
import { t } from '@/locales'
import { Search, Plus, Delete, Check, Close } from '@element-plus/icons-vue'

const listQuery = reactive({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  status: ''
})
const list = ref<CashTransfer[]>([])
const total = ref(0)
const listLoading = ref(true)

const getList = async () => {
  listLoading.value = true
  try {
    const params: Record<string, any> = {}
    if (listQuery.status) params.status = listQuery.status
    
    const response = await listCashTransfersAPI(params)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load transfer list:', error)
  }
}

const handleSearch = () => {
  listQuery.pageNum = 1
  getList()
}

const handleReset = () => {
  listQuery.status = ''
  listQuery.pageNum = 1
  getList()
}

const handleAdd = () => {
  ElMessage.info('Create page not implemented yet')
}

const handleSubmit = async (row: CashTransfer) => {
  try {
    await submitCashTransferAPI(row.id)
    ElMessage.success('Submitted successfully')
    getList()
  } catch (error) {
    console.error('Failed to submit transfer:', error)
  }
}

const handleApprove = async (row: CashTransfer) => {
  try {
    await approveCashTransferAPI(row.id, 1, 'Admin')
    ElMessage.success('Approved successfully')
    getList()
  } catch (error) {
    console.error('Failed to approve transfer:', error)
  }
}

const handleReject = async (row: CashTransfer) => {
  try {
    await rejectCashTransferAPI(row.id)
    ElMessage.success('Rejected successfully')
    getList()
  } catch (error) {
    console.error('Failed to reject transfer:', error)
  }
}

const handleCancel = async (row: CashTransfer) => {
  try {
    await cancelCashTransferAPI(row.id)
    ElMessage.success('Cancelled successfully')
    getList()
  } catch (error) {
    console.error('Failed to cancel transfer:', error)
  }
}

const formatMoney = (value: number | undefined | null) => {
  if (value === undefined || value === null) return '0.00'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
}

const getStatusLabel = (status: string) => {
  const statusMap: Record<string, string> = {
    DRAFT: 'Draft',
    PENDING_APPROVAL: 'Pending',
    APPROVED: 'Approved',
    REJECTED: 'Rejected',
    EXECUTED: 'Executed',
    CANCELLED: 'Cancelled'
  }
  return statusMap[status] || status
}

const getStatusType = (status: string) => {
  const typeMap: Record<string, string> = {
    DRAFT: 'info',
    PENDING_APPROVAL: 'warning',
    APPROVED: 'success',
    REJECTED: 'danger',
    EXECUTED: 'success',
    CANCELLED: 'info'
  }
  return typeMap[status] || 'info'
}

onMounted(() => {
  getList()
})
</script>

<template>
  <div class="cash-transfer-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('cash.transfer.title') || 'Cash Transfer Management' }}</span>
          <el-button type="primary" @click="handleAdd">
            <el-icon><Plus /></el-icon>
            {{ t('common.add') }}
          </el-button>
        </div>
      </template>
      
      <el-table v-loading="listLoading" :data="list" border style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="transferNo" label="Transfer No." width="180" />
        <el-table-column prop="fromAccountName" label="From Account" min-width="150" />
        <el-table-column prop="toAccountName" label="To Account" min-width="150" />
        <el-table-column prop="amount" label="Amount" width="150">
          <template #default="{ row }">
            ¥{{ formatMoney(row.amount) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="Status" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" size="small">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="Created At" width="180" />
        <el-table-column fixed="right" label="Operations" width="200">
          <template #default="{ row }">
            <el-button v-if="row.status === 'DRAFT'" link type="success" size="small" @click="handleSubmit(row)">Submit</el-button>
            <el-button v-if="row.status === 'PENDING_APPROVAL'" link type="success" size="small" @click="handleApprove(row)">Approve</el-button>
            <el-button v-if="row.status === 'PENDING_APPROVAL'" link type="danger" size="small" @click="handleReject(row)">Reject</el-button>
            <el-button v-if="row.status === 'DRAFT'" link type="danger" size="small" @click="handleCancel(row)">Cancel</el-button>
            <el-button v-if="row.status === 'DRAFT'" link type="danger" size="small" @click="handleDelete(row)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.cash-transfer-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>