<template>
  <div class="reporting-container">
    <div class="header">
      <h2>财务报表</h2>
    </div>

    <el-tabs v-model="activeTab" type="border-card" @tab-change="handleTabChange">
      <el-tab-pane label="资产负债表" name="balance">
        <div class="report-toolbar">
          <el-date-picker
            v-model="balanceDate"
            type="datetime"
            placeholder="选择报表日期"
            value-format="YYYY-MM-DD HH:mm:ss"
            @change="loadBalanceSheet"
          />
          <el-button type="primary" @click="loadBalanceSheet">查询</el-button>
          <el-button @click="handleExportBalanceSheet">导出</el-button>
        </div>
        <div v-loading="balanceLoading" class="report-content">
          <div v-if="balanceSheet" class="balance-sheet">
            <h3>资产负债表</h3>
            <p class="report-date">报表日期: {{ balanceSheet.asOfDate }}</p>
            
            <el-row :gutter="20" class="summary-row">
              <el-col :span="8">
                <el-card>
                  <div class="summary-item">
                    <span class="label">资产总计</span>
                    <span class="value">{{ formatNumber(balanceSheet.totalAssets) }}</span>
                  </div>
                </el-card>
              </el-col>
              <el-col :span="8">
                <el-card>
                  <div class="summary-item">
                    <span class="label">负债总计</span>
                    <span class="value">{{ formatNumber(balanceSheet.totalLiabilities) }}</span>
                  </div>
                </el-card>
              </el-col>
              <el-col :span="8">
                <el-card>
                  <div class="summary-item">
                    <span class="label">权益总计</span>
                    <span class="value">{{ formatNumber(balanceSheet.totalEquity) }}</span>
                  </div>
                </el-card>
              </el-col>
            </el-row>

            <el-tabs v-model="balanceSubTab">
              <el-tab-pane label="资产" name="assets">
                <el-table :data="balanceSheet.assets" border size="small">
                  <el-table-column prop="code" label="科目编码" width="120" />
                  <el-table-column prop="name" label="科目名称" min-width="150" />
                  <el-table-column prop="currentAmount" label="本期金额" width="150">
                    <template #default="{ row }">
                      {{ formatNumber(row.currentAmount) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="previousAmount" label="上期金额" width="150">
                    <template #default="{ row }">
                      {{ formatNumber(row.previousAmount) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="change" label="变动金额" width="150">
                    <template #default="{ row }">
                      <span :class="row.change >= 0 ? 'positive' : 'negative'">
                        {{ formatNumber(row.change) }}
                      </span>
                    </template>
                  </el-table-column>
                  <el-table-column prop="changePercent" label="变动比例" width="120">
                    <template #default="{ row }">
                      <span :class="row.changePercent >= 0 ? 'positive' : 'negative'">
                        {{ row.changePercent.toFixed(2) }}%
                      </span>
                    </template>
                  </el-table-column>
                </el-table>
              </el-tab-pane>
              <el-tab-pane label="负债" name="liabilities">
                <el-table :data="balanceSheet.liabilities" border size="small">
                  <el-table-column prop="code" label="科目编码" width="120" />
                  <el-table-column prop="name" label="科目名称" min-width="150" />
                  <el-table-column prop="currentAmount" label="本期金额" width="150">
                    <template #default="{ row }">
                      {{ formatNumber(row.currentAmount) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="previousAmount" label="上期金额" width="150">
                    <template #default="{ row }">
                      {{ formatNumber(row.previousAmount) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="change" label="变动金额" width="150">
                    <template #default="{ row }">
                      <span :class="row.change >= 0 ? 'positive' : 'negative'">
                        {{ formatNumber(row.change) }}
                      </span>
                    </template>
                  </el-table-column>
                  <el-table-column prop="changePercent" label="变动比例" width="120">
                    <template #default="{ row }">
                      <span :class="row.changePercent >= 0 ? 'positive' : 'negative'">
                        {{ row.changePercent.toFixed(2) }}%
                      </span>
                    </template>
                  </el-table-column>
                </el-table>
              </el-tab-pane>
              <el-tab-pane label="所有者权益" name="equity">
                <el-table :data="balanceSheet.equity" border size="small">
                  <el-table-column prop="code" label="科目编码" width="120" />
                  <el-table-column prop="name" label="科目名称" min-width="150" />
                  <el-table-column prop="currentAmount" label="本期金额" width="150">
                    <template #default="{ row }">
                      {{ formatNumber(row.currentAmount) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="previousAmount" label="上期金额" width="150">
                    <template #default="{ row }">
                      {{ formatNumber(row.previousAmount) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="change" label="变动金额" width="150">
                    <template #default="{ row }">
                      <span :class="row.change >= 0 ? 'positive' : 'negative'">
                        {{ formatNumber(row.change) }}
                      </span>
                    </template>
                  </el-table-column>
                  <el-table-column prop="changePercent" label="变动比例" width="120">
                    <template #default="{ row }">
                      <span :class="row.changePercent >= 0 ? 'positive' : 'negative'">
                        {{ row.changePercent.toFixed(2) }}%
                      </span>
                    </template>
                  </el-table-column>
                </el-table>
              </el-tab-pane>
            </el-tabs>
          </div>
          <el-empty v-else description="请选择日期查询报表" />
        </div>
      </el-tab-pane>

      <el-tab-pane label="利润表" name="income">
        <div class="report-toolbar">
          <el-date-picker
            v-model="incomeStartDate"
            type="date"
            placeholder="开始日期"
            value-format="YYYY-MM-DD"
          />
          <span class="separator">至</span>
          <el-date-picker
            v-model="incomeEndDate"
            type="date"
            placeholder="结束日期"
            value-format="YYYY-MM-DD"
          />
          <el-button type="primary" @click="loadIncomeStatement">查询</el-button>
          <el-button @click="handleExportIncomeStatement">导出</el-button>
        </div>
        <div v-loading="incomeLoading" class="report-content">
          <div v-if="incomeStatement" class="income-statement">
            <h3>利润表</h3>
            <p class="report-date">
              报表期间: {{ incomeStatement.startDate }} 至 {{ incomeStatement.endDate }}
            </p>

            <el-row :gutter="20" class="summary-row">
              <el-col :span="6">
                <el-card>
                  <div class="summary-item">
                    <span class="label">营业收入</span>
                    <span class="value positive">{{ formatNumber(incomeStatement.totalRevenue) }}</span>
                  </div>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card>
                  <div class="summary-item">
                    <span class="label">营业成本</span>
                    <span class="value negative">{{ formatNumber(incomeStatement.totalCost) }}</span>
                  </div>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card>
                  <div class="summary-item">
                    <span class="label">毛利润</span>
                    <span class="value">{{ formatNumber(incomeStatement.grossProfit) }}</span>
                  </div>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card>
                  <div class="summary-item">
                    <span class="label">净利润</span>
                    <span class="value">{{ formatNumber(incomeStatement.netProfit) }}</span>
                  </div>
                </el-card>
              </el-col>
            </el-row>

            <el-tabs v-model="incomeSubTab">
              <el-tab-pane label="收入" name="revenue">
                <el-table :data="incomeStatement.revenue" border size="small">
                  <el-table-column prop="code" label="科目编码" width="120" />
                  <el-table-column prop="name" label="科目名称" min-width="150" />
                  <el-table-column prop="amount" label="金额" width="150">
                    <template #default="{ row }">
                      {{ formatNumber(row.amount) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="proportion" label="占比" width="120">
                    <template #default="{ row }">
                      {{ row.proportion.toFixed(2) }}%
                    </template>
                  </el-table-column>
                </el-table>
              </el-tab-pane>
              <el-tab-pane label="成本" name="cost">
                <el-table :data="incomeStatement.cost" border size="small">
                  <el-table-column prop="code" label="科目编码" width="120" />
                  <el-table-column prop="name" label="科目名称" min-width="150" />
                  <el-table-column prop="amount" label="金额" width="150">
                    <template #default="{ row }">
                      {{ formatNumber(row.amount) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="proportion" label="占比" width="120">
                    <template #default="{ row }">
                      {{ row.proportion.toFixed(2) }}%
                    </template>
                  </el-table-column>
                </el-table>
              </el-tab-pane>
              <el-tab-pane label="费用" name="expenses">
                <el-table :data="incomeStatement.expenses" border size="small">
                  <el-table-column prop="code" label="科目编码" width="120" />
                  <el-table-column prop="name" label="科目名称" min-width="150" />
                  <el-table-column prop="amount" label="金额" width="150">
                    <template #default="{ row }">
                      {{ formatNumber(row.amount) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="proportion" label="占比" width="120">
                    <template #default="{ row }">
                      {{ row.proportion.toFixed(2) }}%
                    </template>
                  </el-table-column>
                </el-table>
              </el-tab-pane>
            </el-tabs>
          </div>
          <el-empty v-else description="请选择日期区间查询报表" />
        </div>
      </el-tab-pane>

      <el-tab-pane label="试算平衡表" name="trial">
        <div class="report-toolbar">
          <el-date-picker
            v-model="trialDate"
            type="datetime"
            placeholder="选择报表日期"
            value-format="YYYY-MM-DD HH:mm:ss"
            @change="loadTrialBalance"
          />
          <el-button type="primary" @click="loadTrialBalance">查询</el-button>
          <el-button @click="handleExportTrialBalance">导出</el-button>
        </div>
        <div v-loading="trialLoading" class="report-content">
          <div v-if="trialBalance" class="trial-balance">
            <h3>试算平衡表</h3>
            <p class="report-date">报表日期: {{ trialBalance.asOfDate }}</p>

            <el-alert
              :type="trialBalance.isBalanced ? 'success' : 'error'"
              :title="trialBalance.isBalanced ? '试算平衡' : '试算不平衡'"
              :description="`借方合计: ${formatNumber(trialBalance.totalDebit)} | 贷方合计: ${formatNumber(trialBalance.totalCredit)}`"
              :closable="false"
              show-icon
              style="margin-bottom: 20px"
            />

            <el-table :data="trialBalance.accounts" border size="small" max-height="500">
              <el-table-column prop="subjectCode" label="科目编码" width="120" />
              <el-table-column prop="subjectName" label="科目名称" min-width="150" />
              <el-table-column prop="debitBalance" label="借方余额" width="150" align="right">
                <template #default="{ row }">
                  {{ row.debitBalance > 0 ? formatNumber(row.debitBalance) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="creditBalance" label="贷方余额" width="150" align="right">
                <template #default="{ row }">
                  {{ row.creditBalance > 0 ? formatNumber(row.creditBalance) : '-' }}
                </template>
              </el-table-column>
              <el-table-column prop="debitCredit" label="借贷方向" width="100" align="center">
                <template #default="{ row }">
                  <el-tag :type="row.debitCredit === 'DEBIT' ? 'warning' : 'success'" size="small">
                    {{ row.debitCredit === 'DEBIT' ? '借方' : '贷方' }}
                  </el-tag>
                </template>
              </el-table-column>
            </el-table>
          </div>
          <el-empty v-else description="请选择日期查询报表" />
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import {
  getBalanceSheet,
  getIncomeStatement,
  getTrialBalance,
  exportBalanceSheet,
  exportIncomeStatement,
  exportTrialBalance,
  type BalanceSheetReport,
  type IncomeStatementReport,
  type TrialBalanceReport
} from '@/apis/reporting'

const activeTab = ref('balance')
const balanceSubTab = ref('assets')
const incomeSubTab = ref('revenue')

const balanceDate = ref('')
const balanceLoading = ref(false)
const balanceSheet = ref<BalanceSheetReport | null>(null)

const incomeStartDate = ref('')
const incomeEndDate = ref('')
const incomeLoading = ref(false)
const incomeStatement = ref<IncomeStatementReport | null>(null)

const trialDate = ref('')
const trialLoading = ref(false)
const trialBalance = ref<TrialBalanceReport | null>(null)

const formatNumber = (num: number | undefined | null): string => {
  if (num === undefined || num === null) return '0.00'
  return new Intl.NumberFormat('zh-CN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(num)
}

const handleTabChange = (tabName: any) => {
  if (tabName === 'balance' && balanceSheet.value) {
    balanceSubTab.value = 'assets'
  } else if (tabName === 'income' && incomeStatement.value) {
    incomeSubTab.value = 'revenue'
  }
}

const loadBalanceSheet = async () => {
  if (!balanceDate.value) {
    ElMessage.warning('请选择报表日期')
    return
  }
  balanceLoading.value = true
  try {
    const res = await getBalanceSheet(balanceDate.value)
    if (res.data.code === 0) {
      balanceSheet.value = res.data.data
    } else {
      ElMessage.error(res.data.message || '查询失败')
    }
  } catch (error) {
    ElMessage.error('查询资产负债表失败')
  } finally {
    balanceLoading.value = false
  }
}

const loadIncomeStatement = async () => {
  if (!incomeStartDate.value || !incomeEndDate.value) {
    ElMessage.warning('请选择日期区间')
    return
  }
  incomeLoading.value = true
  try {
    const res = await getIncomeStatement(incomeStartDate.value, incomeEndDate.value)
    if (res.data.code === 0) {
      incomeStatement.value = res.data.data
    } else {
      ElMessage.error(res.data.message || '查询失败')
    }
  } catch (error) {
    ElMessage.error('查询利润表失败')
  } finally {
    incomeLoading.value = false
  }
}

const loadTrialBalance = async () => {
  if (!trialDate.value) {
    ElMessage.warning('请选择报表日期')
    return
  }
  trialLoading.value = true
  try {
    const res = await getTrialBalance(trialDate.value)
    if (res.data.code === 0) {
      trialBalance.value = res.data.data
    } else {
      ElMessage.error(res.data.message || '查询失败')
    }
  } catch (error) {
    ElMessage.error('查询试算平衡表失败')
  } finally {
    trialLoading.value = false
  }
}

const handleExportBalanceSheet = async () => {
  if (!balanceDate.value) {
    ElMessage.warning('请选择报表日期')
    return
  }
  try {
    const res = await exportBalanceSheet(balanceDate.value)
    const blob = new Blob([res as unknown as Blob], { type: 'application/vnd.ms-excel' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `资产负债表_${balanceDate.value}.xlsx`
    link.click()
    window.URL.revokeObjectURL(url)
    ElMessage.success('导出成功')
  } catch (error) {
    ElMessage.error('导出失败')
  }
}

const handleExportIncomeStatement = async () => {
  if (!incomeStartDate.value || !incomeEndDate.value) {
    ElMessage.warning('请选择日期区间')
    return
  }
  try {
    const res = await exportIncomeStatement(incomeStartDate.value, incomeEndDate.value)
    const blob = new Blob([res as unknown as Blob], { type: 'application/vnd.ms-excel' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `利润表_${incomeStartDate.value}_${incomeEndDate.value}.xlsx`
    link.click()
    window.URL.revokeObjectURL(url)
    ElMessage.success('导出成功')
  } catch (error) {
    ElMessage.error('导出失败')
  }
}

const handleExportTrialBalance = async () => {
  if (!trialDate.value) {
    ElMessage.warning('请选择报表日期')
    return
  }
  try {
    const res = await exportTrialBalance(trialDate.value)
    const blob = new Blob([res as unknown as Blob], { type: 'application/vnd.ms-excel' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `试算平衡表_${trialDate.value}.xlsx`
    link.click()
    window.URL.revokeObjectURL(url)
    ElMessage.success('导出成功')
  } catch (error) {
    ElMessage.error('导出失败')
  }
}
</script>

<style scoped>
.reporting-container {
  padding: 20px;
}

.header h2 {
  margin: 0 0 20px 0;
  font-size: 20px;
  font-weight: 600;
}

.report-toolbar {
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.report-toolbar .separator {
  color: #909399;
}

.report-content {
  min-height: 400px;
}

.balance-sheet h3,
.income-statement h3,
.trial-balance h3 {
  text-align: center;
  margin-bottom: 10px;
}

.report-date {
  text-align: center;
  color: #909399;
  margin-bottom: 20px;
}

.summary-row {
  margin-bottom: 20px;
}

.summary-item {
  text-align: center;
}

.summary-item .label {
  display: block;
  color: #909399;
  font-size: 14px;
  margin-bottom: 8px;
}

.summary-item .value {
  font-size: 20px;
  font-weight: 600;
  color: #303133;
}

.summary-item .value.positive {
  color: #67c23a;
}

.summary-item .value.negative {
  color: #f56c6c;
}

.positive {
  color: #67c23a;
}

.negative {
  color: #f56c6c;
}
</style>