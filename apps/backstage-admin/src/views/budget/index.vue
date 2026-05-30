<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  listBudgetsAPI,
  deleteBudgetAPI,
  submitBudgetAPI,
  approveBudgetAPI,
  rejectBudgetAPI,
  closeBudgetAPI
} from '@/apis/budget'
import { Search, Plus, Edit, Delete, Check, Close, Lock } from '@element-plus/icons-vue'
import type { Budget } from '@/apis/budget'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { t } from '@/locales'

const router = useRouter()

const listQuery = reactive({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  departmentId: undefined as number | undefined,
  status: '',
  period: ''
})
const list = ref<Budget[]>([])
const total = ref(0)
const listLoading = ref(true)

const statusOptions = [
  { label: 'Draft', value: 'DRAFT' },
  { label: 'Pending Approval', value: 'PENDING_APPROVAL' },
  { label: 'Approved', value: 'APPROVED' },
  { label: 'Rejected', value: 'REJECTED' },
  { label: 'Closed', value: 'CLOSED' }
]

const periodOptions = [
  { label: 'FY2025', value: 'FY2025' },
  { label: 'FY2026', value: 'FY2026' },
  { label: 'FY2027', value: 'FY2027' },
  { label: 'Q1 2025', value: 'Q1_2025' },
  { label: 'Q2 2025', value: 'Q2_2025' },
  { label: 'Q3 2025', value: 'Q3_2025' },
  { label: 'Q4 2025', value: 'Q4_2025' },
  { label: 'Q1 2026', value: 'Q1_2026' },
  { label: 'Q2 2026', value: 'Q2_2026' }
]

const typeOptions = [
  { label: 'Annual', value: 'ANNUAL' },
  { label: 'Quarterly', value: 'QUARTERLY' },
  { label: 'Monthly', value: 'MONTHLY' },
  { label: 'Project', value: 'PROJECT' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const params: Record<string, any> = {}
    if (listQuery.departmentId) params.departmentId = listQuery.departmentId
    if (listQuery.status) params.status = listQuery.status
    if (listQuery.period) params.period = listQuery.period
    
    const response = await listBudgetsAPI(params)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load budget list:', error)
  }
}

const handleSearch = () => {
  listQuery.pageNum = 1
  getList()
}

const handleReset = () => {
  listQuery.departmentId = undefined
  listQuery.status = ''
  listQuery.period = ''
  listQuery.pageNum = 1
  getList()
}

const handleAdd = () => {
  router.push('/budget/create')
}

const handleEdit = (row: Budget) => {
  router.push(`/budget/edit/${row.id}`)
}

const handleView = (row: Budget) => {
  router.push(`/budget/detail/${row.id}`)
}

const handleDelete = async (row: Budget) => {
  try {
    await ElMessageBox.confirm(
      t('dialogs.deleteConfirm'),
      t('dialogs.warning'),
      { type: 'warning' }
    )
    await deleteBudgetAPI(row.id)
    ElMessage.success(t('messages.deleteSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(error.message || t('messages.deleteFailed'))
    }
  }
}

const handleSubmit = async (row: Budget) => {
  try {
    await submitBudgetAPI(row.id)
    ElMessage.success(t('messages.operationSuccess'))
    getList()
  } catch (error: any) {
    ElMessage.error(error.message || t('messages.operationFailed'))
  }
}

const handleApprove = async (row: Budget) => {
  try {
    await approveBudgetAPI(row.id, 1, 'Admin')
    ElMessage.success(t('messages.operationSuccess'))
    getList()
  } catch (error: any) {
    ElMessage.error(error.message || t('messages.operationFailed'))
  }
}

const handleReject = async (row: Budget) => {
  try {
    await rejectBudgetAPI(row.id)
    ElMessage.success(t('messages.operationSuccess'))
    getList()
  } catch (error: any) {
    ElMessage.error(error.message || t('messages.operationFailed'))
  }
}

const handleClose = async (row: Budget) => {
  try {
    await closeBudgetAPI(row.id)
    ElMessage.success(t('messages.operationSuccess'))
    getList()
  } catch (error: any) {
    ElMessage.error(error.message || t('messages.operationFailed'))
  }
}

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' | undefined => {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    DRAFT: 'info',
    PENDING_APPROVAL: 'warning',
    APPROVED: 'success',
    REJECTED: 'danger',
    CLOSED: 'info'
  }
  return map[status] || 'info'
}

const formatAmount = (amount: number | undefined) => {
  if (amount === undefined || amount === null) return '-'
  return new Intl.NumberFormat('zh-CN', { style: 'currency', currency: 'CNY' }).format(amount)
}

const formatDate = (date: string | null | undefined) => {
  if (!date) return '-'
  return new Date(date).toLocaleDateString()
}

const handleSizeChange = (val: number) => {
  listQuery.pageSize = val
  getList()
}

const handleCurrentChange = (val: number) => {
  listQuery.pageNum = val
  getList()
}

onMounted(() => {
  getList()
})
</script>

<template>
  <div class="budget-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item :label="t('fields.status')">
          <el-select v-model="listQuery.status" :placeholder="t('placeholders.select')" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('fields.period')">
          <el-select v-model="listQuery.period" :placeholder="t('placeholders.select')" clearable>
            <el-option v-for="item in periodOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">{{ t('buttons.search') }}</el-button>
          <el-button @click="handleReset">{{ t('buttons.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <div class="table-header">
        <el-button type="primary" :icon="Plus" @click="handleAdd">{{ t('buttons.create') }}</el-button>
      </div>

      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column prop="budgetCode" :label="t('fields.budgetCode')" width="150" />
        <el-table-column prop="budgetName" :label="t('fields.budgetName')" min-width="150" />
        <el-table-column prop="type" :label="t('fields.type')" width="100">
          <template #default="{ row }">
            {{ typeOptions.find(t => t.value === row.type)?.label || row.type }}
          </template>
        </el-table-column>
        <el-table-column prop="period" :label="t('fields.period')" width="100" />
        <el-table-column prop="departmentName" :label="t('fields.department')" width="120" />
        <el-table-column prop="totalAmount" :label="t('fields.totalAmount')" width="140">
          <template #default="{ row }">
            {{ formatAmount(row.totalAmount) }}
          </template>
        </el-table-column>
        <el-table-column prop="usedAmount" :label="t('fields.usedAmount')" width="140">
          <template #default="{ row }">
            {{ formatAmount(row.usedAmount) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" :label="t('fields.status')" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ statusOptions.find(s => s.value === row.status)?.label || row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" :label="t('fields.createdAt')" width="120">
          <template #default="{ row }">
            {{ formatDate(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('fields.actions')" width="240" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('buttons.view') }}</el-button>
            <el-button v-if="row.status === 'DRAFT'" link type="primary" size="small" @click="handleEdit(row)">{{ t('buttons.edit') }}</el-button>
            <el-button v-if="row.status === 'DRAFT'" link type="success" size="small" @click="handleSubmit(row)">{{ t('buttons.submit') }}</el-button>
            <el-button v-if="row.status === 'PENDING_APPROVAL'" link type="success" size="small" @click="handleApprove(row)">{{ t('buttons.approve') }}</el-button>
            <el-button v-if="row.status === 'PENDING_APPROVAL'" link type="danger" size="small" @click="handleReject(row)">{{ t('buttons.reject') }}</el-button>
            <el-button v-if="row.status === 'APPROVED'" link type="info" size="small" @click="handleClose(row)">{{ t('buttons.close') }}</el-button>
            <el-button v-if="row.status === 'DRAFT' || row.status === 'REJECTED'" link type="danger" size="small" @click="handleDelete(row)">{{ t('buttons.delete') }}</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="listQuery.pageNum"
        v-model:page-size="listQuery.pageSize"
        :page-sizes="PAGE_SIZE_OPTIONS"
        :total="total"
        layout="total, sizes, prev, pager, next, jumper"
        class="pagination"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </el-card>
  </div>
</template>

<style scoped>
.budget-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}

.search-form {
  margin-bottom: 0;
}

.table-card {
  min-height: 500px;
}

.table-header {
  margin-bottom: 16px;
}

.pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>