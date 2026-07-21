<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Refresh } from '@element-plus/icons-vue'
import {
  getCommissionBalanceAPI,
  getCommissionLedgerAPI,
  bindCommissionRelationAPI,
  calculateCommissionAPI,
  settleCommissionAPI,
  cancelCommissionAPI,
  type CommissionAccount,
  type CommissionRecord,
  type CommissionLedgerQuery,
} from '@/apis/commission'

const activeTab = ref('ledger')
const userId = ref(0)
const balance = ref<CommissionAccount | null>(null)

const ledgerQuery = ref<CommissionLedgerQuery>({ userId: 0, page: 1, size: 10 })
const ledger = ref<CommissionRecord[]>([])

const bindDialog = ref(false)
const bindUserId = ref(0)
const bindParentUserId = ref(0)

const settleDialog = ref(false)
const executeBefore = ref('')

const calcDialog = ref(false)
const calcForm = ref({ orderId: 0, userId: 0, payAmount: 0, availableAt: '' })

const loadBalance = async () => {
  if (!userId.value) return
  try {
    const res = await getCommissionBalanceAPI(userId.value)
    balance.value = res.data
  } catch (error) {
    console.error('Failed to load balance:', error)
  }
}

const loadLedger = async () => {
  if (!ledgerQuery.value.userId) return
  try {
    const res = await getCommissionLedgerAPI(ledgerQuery.value)
    ledger.value = res.data || []
  } catch (error) {
    console.error('Failed to load ledger:', error)
  }
}

const queryByUser = () => {
  ledgerQuery.value.userId = userId.value
  loadBalance()
  loadLedger()
}

const handleBind = async () => {
  try {
    await bindCommissionRelationAPI({ userId: bindUserId.value, parentUserId: bindParentUserId.value })
    ElMessage.success('Relation bound')
    bindDialog.value = false
  } catch (error) {
    ElMessage.error('Failed to bind relation')
  }
}

const handleCalc = async () => {
  try {
    await calculateCommissionAPI(calcForm.value)
    ElMessage.success('Commission calculated')
    calcDialog.value = false
  } catch (error) {
    ElMessage.error('Failed to calculate commission')
  }
}

const handleSettle = async () => {
  try {
    await settleCommissionAPI({ executeBefore: executeBefore.value })
    ElMessage.success('Settlement completed')
    settleDialog.value = false
  } catch (error) {
    ElMessage.error('Failed to settle')
  }
}

const handleCancel = async (row: CommissionRecord) => {
  try {
    await ElMessageBox.confirm(`Cancel commission for order ${row.orderId}?`, 'Confirm', {
      confirmButtonText: 'Confirm', cancelButtonText: 'Cancel', type: 'warning',
    })
    await cancelCommissionAPI({ orderId: row.orderId })
    ElMessage.success('Commission cancelled')
    loadLedger()
  } catch (error) {
    if (error !== 'cancel') ElMessage.error('Failed to cancel')
  }
}

const statusTag = (status?: string): 'success' | 'warning' | 'info' | 'danger' => {
  const map: Record<string, 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING: 'warning', SETTLED: 'success', CANCELLED: 'info',
  }
  return map[status || ''] || 'info'
}
</script>

<template>
  <div class="commission-container">
    <el-card class="query-card">
      <el-form :inline="true">
        <el-form-item label="User ID">
          <el-input-number v-model="userId" :min="1" style="width: 160px" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="queryByUser">Query</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-row :gutter="20" class="stat-row">
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-value">{{ balance?.totalAmount?.toFixed(2) || '-' }}</div>
          <div class="stat-label">Total Commission</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-value">{{ balance?.availableAmount?.toFixed(2) || '-' }}</div>
          <div class="stat-label">Available</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-value">{{ balance?.frozenAmount?.toFixed(2) || '-' }}</div>
          <div class="stat-label">Frozen</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div style="display: flex; gap: 8px; justify-content: center;">
            <el-button size="small" @click="bindDialog = true">Bind</el-button>
            <el-button size="small" @click="calcDialog = true">Calc</el-button>
            <el-button size="small" @click="settleDialog = true">Settle</el-button>
          </div>
          <div class="stat-label">Actions</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card class="table-card">
      <el-table :data="ledger" border stripe>
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="orderId" label="Order ID" width="90" />
        <el-table-column prop="userId" label="User ID" width="80" />
        <el-table-column prop="parentUserId" label="Parent User" width="100" />
        <el-table-column prop="amount" label="Amount" width="100">
          <template #default="{ row }">¥{{ row.amount?.toFixed(2) }}</template>
        </el-table-column>
        <el-table-column prop="commissionLevel" label="Level" width="80" />
        <el-table-column prop="type" label="Type" width="100" />
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }"><el-tag :type="statusTag(row.status)">{{ row.status }}</el-tag></template>
        </el-table-column>
        <el-table-column prop="createTime" label="Created" width="170">
          <template #default="{ row }">{{ row.createTime || '-' }}</template>
        </el-table-column>
        <el-table-column label="Actions" width="120" fixed="right">
          <template #default="{ row }">
            <el-button v-if="row.status === 'PENDING'" link type="danger" size="small" @click="handleCancel(row)">Cancel</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="bindDialog" title="Bind Relation" width="400px">
      <el-form label-width="130px">
        <el-form-item label="User ID"><el-input-number v-model="bindUserId" :min="1" style="width: 100%" /></el-form-item>
        <el-form-item label="Parent User ID"><el-input-number v-model="bindParentUserId" :min="1" style="width: 100%" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="bindDialog = false">Cancel</el-button>
        <el-button type="primary" @click="handleBind">Bind</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="calcDialog" title="Calculate Commission" width="450px">
      <el-form label-width="130px">
        <el-form-item label="Order ID"><el-input-number v-model="calcForm.orderId" :min="1" style="width: 100%" /></el-form-item>
        <el-form-item label="User ID"><el-input-number v-model="calcForm.userId" :min="1" style="width: 100%" /></el-form-item>
        <el-form-item label="Pay Amount"><el-input-number v-model="calcForm.payAmount" :min="0" :precision="2" style="width: 100%" /></el-form-item>
        <el-form-item label="Available At"><el-input v-model="calcForm.availableAt" placeholder="YYYY-MM-DD HH:mm" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="calcDialog = false">Cancel</el-button>
        <el-button type="primary" @click="handleCalc">Calculate</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="settleDialog" title="Settle Commissions" width="400px">
      <el-form label-width="130px">
        <el-form-item label="Execute Before">
          <el-input v-model="executeBefore" placeholder="YYYY-MM-DD HH:mm" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="settleDialog = false">Cancel</el-button>
        <el-button type="primary" @click="handleSettle">Settle</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.commission-container { padding: 20px; }
.query-card { margin-bottom: 20px; }
.stat-row { margin-bottom: 20px; }
.stat-value { font-size: 24px; font-weight: bold; color: #303133; text-align: center; }
.stat-label { font-size: 13px; color: #909399; text-align: center; margin-top: 8px; }
.table-card { margin-bottom: 20px; }
</style>
