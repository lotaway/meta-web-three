<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listBankAccountsAPI, deleteBankAccountAPI, freezeBankAccountAPI, unfreezeBankAccountAPI, closeBankAccountAPI } from '@/apis/cash'
import type { BankAccount } from '@/apis/cash'
import { DEFAULT_PAGE_SIZE } from '@/constants'
import { t } from '@/locales'
import { Search, Plus, Delete, Lock, Unlock } from '@element-plus/icons-vue'

const listQuery = reactive({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  status: ''
})
const list = ref<BankAccount[]>([])
const total = ref(0)
const listLoading = ref(true)

const getList = async () => {
  listLoading.value = true
  try {
    const params: Record<string, any> = {}
    if (listQuery.status) params.status = listQuery.status
    
    const response = await listBankAccountsAPI(params)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load bank account list:', error)
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

const handleFreeze = async (row: BankAccount) => {
  try {
    await freezeBankAccountAPI(row.id)
    ElMessage.success('Frozen successfully')
    getList()
  } catch (error) {
    console.error('Failed to freeze account:', error)
  }
}

const handleUnfreeze = async (row: BankAccount) => {
  try {
    await unfreezeBankAccountAPI(row.id)
    ElMessage.success('Unfrozen successfully')
    getList()
  } catch (error) {
    console.error('Failed to unfreeze account:', error)
  }
}

const handleClose = async (row: BankAccount) => {
  try {
    await closeBankAccountAPI(row.id)
    ElMessage.success('Closed successfully')
    getList()
  } catch (error) {
    console.error('Failed to close account:', error)
  }
}

const handleDelete = async (row: BankAccount) => {
  try {
    await ElMessageBox.confirm('Are you sure to delete this account?', 'Warning', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    await deleteBankAccountAPI(row.id)
    ElMessage.success('Deleted successfully')
    getList()
  } catch (error) {
    console.error('Failed to delete account:', error)
  }
}

const formatMoney = (value: number | undefined | null) => {
  if (value === undefined || value === null) return '0.00'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
}

const getStatusLabel = (status: string) => {
  const statusMap: Record<string, string> = {
    ACTIVE: 'Active',
    FROZEN: 'Frozen',
    CLOSED: 'Closed'
  }
  return statusMap[status] || status
}

const getStatusType = (status: string) => {
  const typeMap: Record<string, string> = {
    ACTIVE: 'success',
    FROZEN: 'warning',
    CLOSED: 'info'
  }
  return typeMap[status] || 'info'
}

onMounted(() => {
  getList()
})
</script>

<template>
  <div class="bank-account-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('cash.account.title') || 'Bank Account Management' }}</span>
          <el-button type="primary" @click="handleAdd">
            <el-icon><Plus /></el-icon>
            {{ t('common.add') }}
          </el-button>
        </div>
      </template>
      
      <el-table v-loading="listLoading" :data="list" border style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="accountCode" label="Account Code" width="150" />
        <el-table-column prop="accountName" label="Account Name" min-width="150" />
        <el-table-column prop="bankName" label="Bank" width="150" />
        <el-table-column prop="accountNumber" label="Account Number" width="180" />
        <el-table-column prop="balance" label="Balance" width="150">
          <template #default="{ row }">
            ¥{{ formatMoney(row.balance) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" size="small">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column fixed="right" label="Operations" width="200">
          <template #default="{ row }">
            <el-button v-if="row.status === 'ACTIVE'" link type="warning" size="small" @click="handleFreeze(row)">Freeze</el-button>
            <el-button v-if="row.status === 'FROZEN'" link type="success" size="small" @click="handleUnfreeze(row)">Unfreeze</el-button>
            <el-button v-if="row.status !== 'CLOSED'" link type="info" size="small" @click="handleClose(row)">Close</el-button>
            <el-button v-if="row.status === 'CLOSED'" link type="danger" size="small" @click="handleDelete(row)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.bank-account-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>