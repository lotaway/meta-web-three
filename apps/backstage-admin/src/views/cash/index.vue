<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getCashSummaryAPI, listBankAccountsAPI, listCashTransfersAPI, listBankReconciliationsAPI } from '@/apis/cash'
import type { CashSummary, BankAccount, CashTransfer, BankReconciliation } from '@/apis/cash'
import { MESSAGE_DURATION_SHORT } from '@/constants'
import { t } from '@/locales'

const router = useRouter()

const loading = ref(true)
const activeTab = ref('transfers')
const summary = ref<CashSummary>({
  totalBalance: 0,
  activeAccountCount: 0,
  pendingTransferCount: 0,
  pendingReconciliationCount: 0,
  draftPlanCount: 0
})
const recentTransfers = ref<CashTransfer[]>([])
const recentReconciliations = ref<BankReconciliation[]>([])

const formatMoney = (value: number | undefined | null) => {
  if (value === undefined || value === null) return '0.00'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
}

const getSummary = async () => {
  try {
    const response = await getCashSummaryAPI()
    summary.value = response.data || {
      totalBalance: 0,
      activeAccountCount: 0,
      pendingTransferCount: 0,
      pendingReconciliationCount: 0,
      draftPlanCount: 0
    }
  } catch (error) {
    console.error('Failed to load cash summary:', error)
  }
}

const getRecentTransfers = async () => {
  try {
    const response = await listCashTransfersAPI({ status: 'PENDING_APPROVAL' })
    recentTransfers.value = (response.data || []).slice(0, 5)
  } catch (error) {
    console.error('Failed to load recent transfers:', error)
  }
}

const getRecentReconciliations = async () => {
  try {
    const response = await listBankReconciliationsAPI({ status: 'PENDING_APPROVAL' })
    recentReconciliations.value = (response.data || []).slice(0, 5)
  } catch (error) {
    console.error('Failed to load recent reconciliations:', error)
  }
}

const handleNavigation = (path: string) => {
  router.push(path)
}

onMounted(async () => {
  loading.value = true
  await Promise.all([
    getSummary(),
    getRecentTransfers(),
    getRecentReconciliations()
  ])
  loading.value = false
})

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

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const typeMap: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    DRAFT: 'info',
    PENDING_APPROVAL: 'warning',
    APPROVED: 'success',
    REJECTED: 'danger',
    EXECUTED: 'success',
    CANCELLED: 'info'
  }
  return typeMap[status] || 'info'
}
</script>

<template>
  <div class="cash-container">
    <el-row :gutter="20" class="summary-cards">
      <el-col :span="6">
        <el-card class="summary-card" shadow="hover">
          <div class="card-content" @click="handleNavigation('/cash/account')">
            <div class="card-icon total-balance">
              <el-icon :size="32"><Money /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-label">{{ t('cash.totalBalance') }}</div>
              <div class="card-value">¥{{ formatMoney(summary.totalBalance) }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="summary-card" shadow="hover">
          <div class="card-content" @click="handleNavigation('/cash/account')">
            <div class="card-icon account-count">
              <el-icon :size="32"><Wallet /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-label">{{ t('cash.activeAccounts') }}</div>
              <div class="card-value">{{ summary.activeAccountCount }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="summary-card" shadow="hover">
          <div class="card-content" @click="handleNavigation('/cash/transfer')">
            <div class="card-icon pending-transfer">
              <el-icon :size="32"><RefreshRight /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-label">{{ t('cash.pendingTransfers') }}</div>
              <div class="card-value">{{ summary.pendingTransferCount }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="summary-card" shadow="hover">
          <div class="card-content" @click="handleNavigation('/cash/reconciliation')">
            <div class="card-icon pending-reconciliation">
              <el-icon :size="32"><DocumentChecked /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-label">{{ t('cash.pendingReconciliations') }}</div>
              <div class="card-value">{{ summary.pendingReconciliationCount }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="mt-20">
      <el-col :span="12">
        <el-card class="content-card">
          <template #header>
            <div class="card-header">
              <span>{{ t('cash.quickActions') }}</span>
            </div>
          </template>
          <div class="quick-actions">
            <el-button type="primary" @click="handleNavigation('/cash/plan')">
              <el-icon><DocumentAdd /></el-icon>
              {{ t('cash.createPlan') }}
            </el-button>
            <el-button type="success" @click="handleNavigation('/cash/account')">
              <el-icon><Plus /></el-icon>
              {{ t('cash.addAccount') }}
            </el-button>
            <el-button type="warning" @click="handleNavigation('/cash/transfer')">
              <el-icon><Switch /></el-icon>
              {{ t('cash.createTransfer') }}
            </el-button>
            <el-button type="info" @click="handleNavigation('/cash/reconciliation')">
              <el-icon><DocumentChecked /></el-icon>
              {{ t('cash.createReconciliation') }}
            </el-button>
            <el-button @click="handleNavigation('/cash/forecast')">
              <el-icon><TrendCharts /></el-icon>
              {{ t('cash.createForecast') }}
            </el-button>
          </div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card class="content-card">
          <template #header>
            <div class="card-header">
              <span>{{ t('cash.pendingApprovals') }}</span>
            </div>
          </template>
          <el-tabs v-model="activeTab">
            <el-tab-pane :label="t('cash.transfers')" name="transfers">
              <div v-if="recentTransfers.length === 0" class="empty-tip">
                {{ t('cash.noPendingTransfers') }}
              </div>
              <el-table v-else :data="recentTransfers" style="width: 100%">
                <el-table-column prop="transferNo" :label="t('cash.transferNo')" width="180" />
                <el-table-column prop="amount" :label="t('cash.amount')" width="120">
                  <template #default="{ row }">
                    ¥{{ formatMoney(row.amount) }}
                  </template>
                </el-table-column>
                <el-table-column prop="status" :label="t('cash.status')" width="100">
                  <template #default="{ row }">
                    <el-tag :type="getStatusType(row.status)" size="small">
                      {{ getStatusLabel(row.status) }}
                    </el-tag>
                  </template>
                </el-table-column>
              </el-table>
            </el-tab-pane>
            <el-tab-pane :label="t('cash.reconciliations')" name="reconciliations">
              <div v-if="recentReconciliations.length === 0" class="empty-tip">
                {{ t('cash.noPendingReconciliations') }}
              </div>
              <el-table v-else :data="recentReconciliations" style="width: 100%">
                <el-table-column prop="reconciliationNo" :label="t('cash.reconciliationNo')" width="180" />
                <el-table-column prop="bankAccountName" :label="t('cash.accountName')" width="120" />
                <el-table-column prop="status" :label="t('cash.status')" width="100">
                  <template #default="{ row }">
                    <el-tag :type="getStatusType(row.status)" size="small">
                      {{ getStatusLabel(row.status) }}
                    </el-tag>
                  </template>
                </el-table-column>
              </el-table>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.cash-container {
  padding: 20px;
}

.summary-cards {
  margin-bottom: 20px;
}

.summary-card {
  cursor: pointer;
  transition: transform 0.2s;
}

.summary-card:hover {
  transform: translateY(-4px);
}

.card-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.card-icon {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
}

.total-balance {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.account-count {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.pending-transfer {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.pending-reconciliation {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

.card-info {
  flex: 1;
}

.card-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 4px;
}

.card-value {
  font-size: 24px;
  font-weight: 600;
  color: #303133;
}

.mt-20 {
  margin-top: 20px;
}

.content-card {
  height: 100%;
}

.card-header {
  font-weight: 600;
  font-size: 16px;
}

.quick-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.empty-tip {
  text-align: center;
  color: #909399;
  padding: 40px 0;
}
</style>