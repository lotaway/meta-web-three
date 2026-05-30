<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listCashPlansAPI, deleteCashPlanAPI, submitCashPlanAPI, approveCashPlanAPI, rejectCashPlanAPI } from '@/apis/cash'
import type { CashPlan } from '@/apis/cash'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE } from '@/constants'
import { t } from '@/locales'
import { Search, Plus, Edit, Delete, Check, Close } from '@element-plus/icons-vue'

const listQuery = reactive({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  status: ''
})
const list = ref<CashPlan[]>([])
const total = ref(0)
const listLoading = ref(true)

const statusOptions = [
  { label: 'Draft', value: 'DRAFT' },
  { label: 'Pending Approval', value: 'PENDING_APPROVAL' },
  { label: 'Approved', value: 'APPROVED' },
  { label: 'Rejected', value: 'REJECTED' }
]

const periodOptions = [
  { label: 'Annual', value: 'ANNUAL' },
  { label: 'Quarterly', value: 'QUARTERLY' },
  { label: 'Monthly', value: 'MONTHLY' }
]

const typeOptions = [
  { label: 'Inflow', value: 'INFLOW' },
  { label: 'Outflow', value: 'OUTFLOW' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const params: Record<string, any> = {}
    if (listQuery.status) params.status = listQuery.status
    
    const response = await listCashPlansAPI(params)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load cash plan list:', error)
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
  // TODO: Navigate to create page
  ElMessage.info('Create page not implemented yet')
}

const handleEdit = (row: CashPlan) => {
  // TODO: Navigate to edit page
  ElMessage.info('Edit page not implemented yet')
}

const handleView = (row: CashPlan) => {
  // TODO: Navigate to detail page
  ElMessage.info('Detail page not implemented yet')
}

const handleDelete = async (row: CashPlan) => {
  try {
    await ElMessageBox.confirm('Are you sure to delete this cash plan?', 'Warning', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    await deleteCashPlanAPI(row.id)
    ElMessage.success('Deleted successfully')
    getList()
  } catch (error) {
    console.error('Failed to delete cash plan:', error)
  }
}

const handleSubmit = async (row: CashPlan) => {
  try {
    await submitCashPlanAPI(row.id)
    ElMessage.success('Submitted successfully')
    getList()
  } catch (error) {
    console.error('Failed to submit cash plan:', error)
  }
}

const handleApprove = async (row: CashPlan) => {
  try {
    await approveCashPlanAPI(row.id, 1, 'Admin')
    ElMessage.success('Approved successfully')
    getList()
  } catch (error) {
    console.error('Failed to approve cash plan:', error)
  }
}

const handleReject = async (row: CashPlan) => {
  try {
    await rejectCashPlanAPI(row.id)
    ElMessage.success('Rejected successfully')
    getList()
  } catch (error) {
    console.error('Failed to reject cash plan:', error)
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
    REJECTED: 'Rejected'
  }
  return statusMap[status] || status
}

const getStatusType = (status: string) => {
  const typeMap: Record<string, string> = {
    DRAFT: 'info',
    PENDING_APPROVAL: 'warning',
    APPROVED: 'success',
    REJECTED: 'danger'
  }
  return typeMap[status] || 'info'
}

onMounted(() => {
  getList()
})
</script>

<template>
  <div class="cash-plan-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('cash.plan.title') || 'Cash Plan Management' }}</span>
          <el-button type="primary" @click="handleAdd">
            <el-icon><Plus /></el-icon>
            {{ t('common.add') }}
          </el-button>
        </div>
      </template>
      
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="Select status" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">
            <el-icon><Search /></el-icon>
            {{ t('common.search') }}
          </el-button>
          <el-button @click="handleReset">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>

      <el-table v-loading="listLoading" :data="list" border style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="planCode" label="Plan Code" width="150" />
        <el-table-column prop="planName" label="Plan Name" min-width="150" />
        <el-table-column prop="type" label="Type" width="100" />
        <el-table-column prop="period" label="Period" width="120" />
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
            <el-button link type="primary" size="small" @click="handleView(row)">View</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">Edit</el-button>
            <el-button v-if="row.status === 'DRAFT'" link type="success" size="small" @click="handleSubmit(row)">Submit</el-button>
            <el-button v-if="row.status === 'PENDING_APPROVAL'" link type="success" size="small" @click="handleApprove(row)">Approve</el-button>
            <el-button v-if="row.status === 'PENDING_APPROVAL'" link type="danger" size="small" @click="handleReject(row)">Reject</el-button>
            <el-button v-if="row.status === 'DRAFT'" link type="danger" size="small" @click="handleDelete(row)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.cash-plan-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.search-form {
  margin-bottom: 20px;
}
</style>