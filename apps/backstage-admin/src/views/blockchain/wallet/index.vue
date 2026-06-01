<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search, Plus, View, Lock, Unlock, Money } from '@element-plus/icons-vue'
import {
  getWalletsAPI,
  getWalletByIdAPI,
  createWalletAPI,
  depositAPI,
  withdrawAPI,
  freezeWalletAPI,
  activateWalletAPI,
  getWalletTransactionsAPI,
  type Wallet,
  type WalletTransaction,
  type WalletQueryParam
} from '@/apis/wallet'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<WalletQueryParam>({ pageNum: 1, pageSize: 10 })
const list = ref<Wallet[]>([])
const listLoading = ref(true)
const total = ref(0)

const detailDialogVisible = ref(false)
const createDialogVisible = ref(false)
const transactionDialogVisible = ref(false)
const dialogLoading = ref(false)

const selectedWallet = ref<Wallet | null>(null)
const transactions = ref<WalletTransaction[]>([])
const transactionLoading = ref(false)

const formData = ref({
  userId: '',
  chainType: 'ETHEREUM',
  address: ''
})

const transactionForm = ref({
  id: 0,
  amount: 0
})

const chainTypeOptions = [
  { label: 'Ethereum', value: 'ETHEREUM' },
  { label: 'Polygon', value: 'POLYGON' },
  { label: 'BSC', value: 'BSC' },
  { label: 'Solana', value: 'SOLANA' },
  { label: 'Polkadot', value: 'POLKADOT' }
]

const statusOptions = [
  { label: 'Active', value: 'ACTIVE' },
  { label: 'Frozen', value: 'FROZEN' },
  { label: 'Closed', value: 'CLOSED' }
]

const transactionTypeOptions = [
  { label: 'Deposit', value: 'DEPOSIT' },
  { label: 'Withdraw', value: 'WITHDRAW' },
  { label: 'Transfer In', value: 'TRANSFER_IN' },
  { label: 'Transfer Out', value: 'TRANSFER_OUT' }
]

const transactionStatusOptions = [
  { label: 'Pending', value: 'PENDING' },
  { label: 'Confirmed', value: 'CONFIRMED' },
  { label: 'Failed', value: 'FAILED' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getWalletsAPI(listQuery.value)
    list.value = response.data || []
    total.value = list.value.length
  } catch (error) {
    ElMessage.error('Failed to load wallets')
    list.value = []
  } finally {
    listLoading.value = false
  }
}

onMounted(() => {
  getList()
})

const handleSearch = () => getList()

const handleReset = () => {
  listQuery.value = { pageNum: 1, pageSize: 10 }
  getList()
}

const handleCreate = async () => {
  if (!formData.value.userId || !formData.value.address) {
    ElMessage.warning('Please fill in required fields')
    return
  }
  dialogLoading.value = true
  try {
    await createWalletAPI(formData.value)
    ElMessage.success('Wallet created successfully')
    createDialogVisible.value = false
    formData.value = { userId: '', chainType: 'ETHEREUM', address: '' }
    getList()
  } catch (error) {
    ElMessage.error('Failed to create wallet')
  } finally {
    dialogLoading.value = false
  }
}

const handleViewDetail = async (row: Wallet) => {
  try {
    const response = await getWalletByIdAPI(row.id)
    selectedWallet.value = response.data
    detailDialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load wallet details')
  }
}

const handleViewTransactions = async (row: Wallet) => {
  selectedWallet.value = row
  transactionLoading.value = true
  transactionDialogVisible.value = true
  try {
    const response = await getWalletTransactionsAPI(row.id)
    transactions.value = response.data || []
  } catch (error) {
    ElMessage.error('Failed to load transactions')
    transactions.value = []
  } finally {
    transactionLoading.value = false
  }
}

const handleDeposit = async () => {
  if (transactionForm.value.amount <= 0) {
    ElMessage.warning('Please enter a valid amount')
    return
  }
  dialogLoading.value = true
  try {
    await depositAPI(transactionForm.value.id, transactionForm.value.amount)
    ElMessage.success('Deposit successful')
    transactionForm.value = { id: 0, amount: 0 }
    getList()
  } catch (error) {
    ElMessage.error('Deposit failed')
  } finally {
    dialogLoading.value = false
  }
}

const handleWithdraw = async () => {
  if (transactionForm.value.amount <= 0) {
    ElMessage.warning('Please enter a valid amount')
    return
  }
  dialogLoading.value = true
  try {
    await withdrawAPI(transactionForm.value.id, transactionForm.value.amount)
    ElMessage.success('Withdrawal successful')
    transactionForm.value = { id: 0, amount: 0 }
    getList()
  } catch (error) {
    ElMessage.error('Withdrawal failed')
  } finally {
    dialogLoading.value = false
  }
}

const handleFreeze = async (row: Wallet) => {
  try {
    await freezeWalletAPI(row.id)
    ElMessage.success('Wallet frozen successfully')
    getList()
  } catch (error) {
    ElMessage.error('Failed to freeze wallet')
  }
}

const handleActivate = async (row: Wallet) => {
  try {
    await activateWalletAPI(row.id)
    ElMessage.success('Wallet activated successfully')
    getList()
  } catch (error) {
    ElMessage.error('Failed to activate wallet')
  }
}

const getStatusLabel = (status: string) => {
  const option = statusOptions.find(s => s.value === status)
  return option?.label || status
}

const getChainTypeLabel = (type: string) => {
  const option = chainTypeOptions.find(c => c.value === type)
  return option?.label || type
}

const getTransactionTypeLabel = (type: string) => {
  const option = transactionTypeOptions.find(t => t.value === type)
  return option?.label || type
}

const getTransactionStatusLabel = (status: string) => {
  const option = transactionStatusOptions.find(s => s.value === status)
  return option?.label || status
}

type StatusTagType = 'info' | 'warning' | 'primary' | 'success' | 'danger'

const getStatusType = (status: string): StatusTagType => {
  const statusMap: Record<string, StatusTagType> = {
    ACTIVE: 'success',
    FROZEN: 'warning',
    CLOSED: 'danger',
    PENDING: 'warning',
    CONFIRMED: 'success',
    FAILED: 'danger'
  }
  return statusMap[status] || 'info'
}

const getChainTypeColor = (type: string): string => {
  const colorMap: Record<string, string> = {
    ETHEREUM: '#627eea',
    POLYGON: '#8247e5',
    BSC: '#f3ba2f',
    SOLANA: '#9945ff',
    POLKADOT: '#e6007a'
  }
  return colorMap[type] || '#666'
}

const openTransactionDialog = (row: Wallet, type: 'deposit' | 'withdraw') => {
  selectedWallet.value = row
  transactionForm.value = { id: row.id, amount: 0 }
  detailDialogVisible.value = false
  createDialogVisible.value = false
  if (type === 'deposit') {
    detailDialogVisible.value = true
  } else {
    detailDialogVisible.value = true
  }
}
</script>

<template>
  <div class="wallet-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item label="User ID">
          <el-input v-model="listQuery.userId" placeholder="Enter user ID" clearable />
        </el-form-item>
        <el-form-item label="Chain Type">
          <el-select v-model="listQuery.chainType" placeholder="Select chain" clearable>
            <el-option v-for="item in chainTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="Select status" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
          <el-button type="success" :icon="Plus" @click="createDialogVisible = true">Create Wallet</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column label="ID" prop="id" width="80" />
        <el-table-column label="User ID" prop="userId" min-width="120" />
        <el-table-column label="Chain" min-width="100">
          <template #default="{ row }">
            <el-tag :style="{ backgroundColor: getChainTypeColor(row.chainType), color: '#fff', border: 'none' }">
              {{ getChainTypeLabel(row.chainType) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Address" prop="address" min-width="200">
          <template #default="{ row }">
            <el-tooltip :content="row.address" placement="top">
              <span class="address-text">{{ row.address.substring(0, 10) }}...{{ row.address.substring(row.address.length - 8) }}</span>
            </el-tooltip>
          </template>
        </el-table-column>
        <el-table-column label="Balance" prop="balance" min-width="120">
          <template #default="{ row }">
            <span class="balance-text">{{ row.balance.toFixed(4) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="Status" min-width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Created At" min-width="160">
          <template #default="{ row }">
            {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="250" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" size="small" :icon="View" @click="handleViewDetail(row)">
              Detail
            </el-button>
            <el-button type="info" size="small" :icon="Money" @click="handleViewTransactions(row)">
              Txs
            </el-button>
            <el-button v-if="row.status === 'ACTIVE'" type="warning" size="small" :icon="Lock" @click="handleFreeze(row)">
              Freeze
            </el-button>
            <el-button v-if="row.status === 'FROZEN'" type="success" size="small" :icon="Unlock" @click="handleActivate(row)">
              Activate
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="createDialogVisible" title="Create Wallet" width="500px">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="User ID" required>
          <el-input v-model="formData.userId" placeholder="Enter user ID" />
        </el-form-item>
        <el-form-item label="Chain Type">
          <el-select v-model="formData.chainType" style="width: 100%">
            <el-option v-for="item in chainTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Address" required>
          <el-input v-model="formData.address" placeholder="Enter wallet address" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleCreate">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="detailDialogVisible" title="Wallet Details" width="600px">
      <div v-if="selectedWallet" class="wallet-detail">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="Wallet ID">{{ selectedWallet.id }}</el-descriptions-item>
          <el-descriptions-item label="User ID">{{ selectedWallet.userId }}</el-descriptions-item>
          <el-descriptions-item label="Chain Type">
            <el-tag :style="{ backgroundColor: getChainTypeColor(selectedWallet.chainType), color: '#fff', border: 'none' }">
              {{ getChainTypeLabel(selectedWallet.chainType) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="Status">
            <el-tag :type="getStatusType(selectedWallet.status)">{{ getStatusLabel(selectedWallet.status) }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="Balance" :span="2">
            <span class="balance-highlight">{{ selectedWallet.balance.toFixed(4) }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="Address" :span="2">
            <el-tooltip :content="selectedWallet.address" placement="top">
              <span class="address-text">{{ selectedWallet.address }}</span>
            </el-tooltip>
          </el-descriptions-item>
          <el-descriptions-item label="Created At">{{ selectedWallet.createdAt ? formatDateTime(selectedWallet.createdAt) : '-' }}</el-descriptions-item>
          <el-descriptions-item label="Updated At">{{ selectedWallet.updatedAt ? formatDateTime(selectedWallet.updatedAt) : '-' }}</el-descriptions-item>
        </el-descriptions>

        <el-divider>Quick Actions</el-divider>

        <div class="action-buttons">
          <el-button type="success" :icon="Money" @click="transactionForm = { id: selectedWallet.id, amount: 0 }; detailDialogVisible = false; createDialogVisible = true">
            Deposit
          </el-button>
          <el-button type="danger" :icon="Money" @click="transactionForm = { id: selectedWallet.id, amount: 0 }; detailDialogVisible = false; createDialogVisible = true">
            Withdraw
          </el-button>
        </div>
      </div>

      <el-dialog v-model="transactionDialogVisible" title="Wallet Transactions" width="700px" append-to-body>
        <el-table v-loading="transactionLoading" :data="transactions" border stripe>
          <el-table-column label="ID" prop="id" width="80" />
          <el-table-column label="Type" min-width="120">
            <template #default="{ row }">
              <el-tag :type="row.type === 'DEPOSIT' ? 'success' : row.type === 'WITHDRAW' ? 'danger' : 'info'">
                {{ getTransactionTypeLabel(row.type) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="Amount" prop="amount" min-width="120">
            <template #default="{ row }">
              <span :class="row.type === 'DEPOSIT' || row.type === 'TRANSFER_IN' ? 'amount-positive' : 'amount-negative'">
                {{ row.type === 'DEPOSIT' || row.type === 'TRANSFER_IN' ? '+' : '-' }}{{ row.amount.toFixed(4) }}
              </span>
            </template>
          </el-table-column>
          <el-table-column label="Fee" prop="fee" min-width="80">
            <template #default="{ row }">
              {{ row.fee ? row.fee.toFixed(4) : '-' }}
            </template>
          </el-table-column>
          <el-table-column label="TxHash" prop="txHash" min-width="150">
            <template #default="{ row }">
              <el-tooltip v-if="row.txHash" :content="row.txHash" placement="top">
                <span class="tx-hash">{{ row.txHash.substring(0, 8) }}...</span>
              </el-tooltip>
              <span v-else>-</span>
            </template>
          </el-table-column>
          <el-table-column label="Status" min-width="100">
            <template #default="{ row }">
              <el-tag :type="getStatusType(row.status)">
                {{ getTransactionStatusLabel(row.status) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="Created At" min-width="160">
            <template #default="{ row }">
              {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
            </template>
          </el-table-column>
        </el-table>

        <template #footer>
          <el-button @click="transactionDialogVisible = false">Close</el-button>
        </template>
      </el-dialog>

      <template #footer>
        <el-button @click="detailDialogVisible = false">Close</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.wallet-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}

.search-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.table-card {
  min-height: 400px;
}

.address-text {
  font-family: monospace;
  font-size: 12px;
}

.balance-text {
  font-weight: 600;
  color: #409eff;
}

.balance-highlight {
  font-size: 20px;
  font-weight: bold;
  color: #67c23a;
}

.wallet-detail {
  padding: 10px;
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 20px;
  padding: 20px 0;
}

.tx-hash {
  font-family: monospace;
  font-size: 12px;
  color: #409eff;
}

.amount-positive {
  color: #67c23a;
  font-weight: 600;
}

.amount-negative {
  color: #f56c6c;
  font-weight: 600;
}
</style>