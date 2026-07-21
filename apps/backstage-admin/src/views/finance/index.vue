<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import {
  listSubjectsAPI, createSubjectAPI, disableSubjectAPI, enableSubjectAPI,
  type AccountSubject,
  listAccountsAPI, createAccountAPI, freezeAccountAPI, unfreezeAccountAPI, closeAccountAPI,
  type Account,
  listVouchersAPI, createVoucherAPI, submitVoucherAPI, approveVoucherAPI, postVoucherAPI,
  type Voucher,
  listArAPI, createArAPI, getOverdueArAPI, listApAPI, createApAPI, getOverdueApAPI,
  type ArApRecord, type CostCenter,
  listCostCentersAPI, listStandardCostsAPI, listActualCostsAPI, listCostVariancesAPI,
  getRatioDashboardAPI, listRatiosAPI,
} from '@/apis/finance'
import { getCashSummaryAPI, listCashPlansAPI, listBankAccountsAPI, listCashTransfersAPI } from '@/apis/cash'
import { listBudgetsAPI } from '@/apis/budget'
import { listAssets } from '@/apis/asset'

const activeTab = ref('dashboard')
const loading = ref(false)

// === Dashboard ===
const dashData = ref<any>({})
const loadDashboard = async () => {
  try {
    const [s, c, b, a, v] = await Promise.allSettled([
      getCashSummaryAPI(), listCashPlansAPI({}),
      listBudgetsAPI({}), listAccountsAPI({}),
      listVouchersAPI({})
    ])
    if (s.status === 'fulfilled') dashData.value.cashSummary = s.value.data
    if (c.status === 'fulfilled') dashData.value.cashPlans = (c.value.data as any)?.length || 0
    if (b.status === 'fulfilled') dashData.value.budgets = (b.value.data as any)?.length || 0
    if (a.status === 'fulfilled') dashData.value.accounts = (a.value.data as any)?.length || 0
    if (v.status === 'fulfilled') dashData.value.vouchers = (v.value.data as any)?.length || 0
  } catch (_) { /* ignore */ }
}

// === Subjects ===
const subjects = ref<AccountSubject[]>([])
const loadSubjects = async () => {
  try { const res = await listSubjectsAPI({}); subjects.value = (res.data as any) || [] } catch (_) { /* ignore */ }
}
const subjectDialog = ref(false)
const subjectForm = ref({ subjectCode: '', subjectName: '', subjectType: 'ASSET', level: 1, balanceDirection: 'DEBIT' })
const createSubject = async () => {
  try { await createSubjectAPI(subjectForm.value); ElMessage.success('Created'); subjectDialog.value = false; loadSubjects() } catch (_) { ElMessage.error('Failed') }
}
const toggleSubject = async (id: number, enable: boolean) => {
  try { enable ? await enableSubjectAPI(id) : await disableSubjectAPI(id); ElMessage.success('Updated'); loadSubjects() } catch (_) { ElMessage.error('Failed') }
}

// === Accounts ===
const accounts = ref<Account[]>([])
const loadFinanceAccounts = async () => {
  try { const res = await listAccountsAPI({}); accounts.value = (res.data as any) || [] } catch (_) { /* ignore */ }
}
const accountDialog = ref(false)
const accountForm = ref({ accountCode: '', accountName: '', accountType: 'CASH', currency: 'CNY', description: '' })
const createFinanceAccount = async () => {
  try { await createAccountAPI(accountForm.value); ElMessage.success('Created'); accountDialog.value = false; loadFinanceAccounts() } catch (_) { ElMessage.error('Failed') }
}

// === Vouchers ===
const vouchers = ref<Voucher[]>([])
const loadVouchers = async () => {
  try { const res = await listVouchersAPI({}); vouchers.value = (res.data as any) || [] } catch (_) { /* ignore */ }
}
const voucherDialog = ref(false)
const voucherForm = ref({ voucherNo: '', voucherType: 'GENERAL', remark: '' })
const createVoucher = async () => {
  try { await createVoucherAPI(voucherForm.value); ElMessage.success('Created'); voucherDialog.value = false; loadVouchers() } catch (_) { ElMessage.error('Failed') }
}
const approveV = async (id: number) => {
  try { await ElMessageBox.prompt('Approver name:', 'Approve').then(async ({ value }) => { if (value) { await approveVoucherAPI(id, value); ElMessage.success('Approved'); loadVouchers() } }) } catch (_) { /* ignore */ }
}
const submitV = async (id: number) => {
  try { await submitVoucherAPI(id); ElMessage.success('Submitted'); loadVouchers() } catch (_) { ElMessage.error('Failed') }
}
const postV = async (id: number) => {
  try { await postVoucherAPI(id); ElMessage.success('Posted'); loadVouchers() } catch (_) { ElMessage.error('Failed') }
}

// === AR/AP ===
const arRecords = ref<ArApRecord[]>([])
const apRecords = ref<ArApRecord[]>([])
const loadArAp = async () => {
  try {
    const [ar, ap, oar, oap] = await Promise.allSettled([
      listArAPI(), listApAPI(), getOverdueArAPI(), getOverdueApAPI()
    ])
    if (ar.status === 'fulfilled') arRecords.value = (ar.value.data as any) || []
    if (ap.status === 'fulfilled') apRecords.value = (ap.value.data as any) || []
    if (oar.status === 'fulfilled') dashData.value.overdueAr = (oar.value.data as any)?.length || 0
    if (oap.status === 'fulfilled') dashData.value.overdueAp = (oap.value.data as any)?.length || 0
  } catch (_) { /* ignore */ }
}
const arDialog = ref(false)
const arForm = ref({ customerOrSupplierName: '', amount: 0, dueDate: '' })
const createAr = async () => {
  try { await createArAPI(arForm.value); ElMessage.success('AR created'); arDialog.value = false; loadArAp() } catch (_) { ElMessage.error('Failed') }
}
const apDialog = ref(false)
const apForm = ref({ customerOrSupplierName: '', amount: 0, dueDate: '' })
const createAp = async () => {
  try { await createApAPI(apForm.value); ElMessage.success('AP created'); apDialog.value = false; loadArAp() } catch (_) { ElMessage.error('Failed') }
}

// === Cost Centers ===
const costCenters = ref<CostCenter[]>([])
const standardCosts = ref<any[]>([])
const actualCosts = ref<any[]>([])
const costVariances = ref<any[]>([])
const loadCost = async () => {
  try {
    const [cc, sc, ac, cv] = await Promise.allSettled([
      listCostCentersAPI(), listStandardCostsAPI(), listActualCostsAPI(), listCostVariancesAPI()
    ])
    if (cc.status === 'fulfilled') costCenters.value = (cc.value.data as any) || []
    if (sc.status === 'fulfilled') standardCosts.value = (sc.value.data as any) || []
    if (ac.status === 'fulfilled') actualCosts.value = (ac.value.data as any) || []
    if (cv.status === 'fulfilled') costVariances.value = (cv.value.data as any) || []
  } catch (_) { /* ignore */ }
}

// === Ratios ===
const ratios = ref<any[]>([])
const ratioDashboard = ref<any>({})
const loadRatios = async () => {
  try {
    const [r, rd] = await Promise.allSettled([listRatiosAPI(), getRatioDashboardAPI()])
    if (r.status === 'fulfilled') ratios.value = (r.value.data as any) || []
    if (rd.status === 'fulfilled') ratioDashboard.value = (rd.value.data as any) || {}
  } catch (_) { /* ignore */ }
}

const refreshAll = () => {
  loadDashboard(); loadSubjects(); loadFinanceAccounts(); loadVouchers(); loadArAp(); loadCost(); loadRatios()
}

const statusTag = (s?: string) => {
  const m: Record<string, string> = { ACTIVE: 'success', INACTIVE: 'info', PENDING: 'info', SUBMITTED: 'warning', APPROVED: 'success', REJECTED: 'danger', POSTED: 'success', CLOSED: 'info', DRAFT: 'info', OVERDUE: 'danger' }
  return m[s || ''] || 'info'
}

onMounted(refreshAll)
</script>

<template>
  <div class="fi-container">
    <div class="toolbar">
      <el-button :icon="Refresh" @click="refreshAll">Refresh</el-button>
    </div>

    <el-tabs v-model="activeTab" type="border-card">
      <!-- Dashboard -->
      <el-tab-pane label="Dashboard" name="dashboard">
        <el-row :gutter="16">
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ dashData.accounts || 0 }}</div><div class="stat-label">Accounts</div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ dashData.vouchers || 0 }}</div><div class="stat-label">Vouchers</div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ dashData.cashPlans || 0 }}</div><div class="stat-label">Cash Plans</div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ dashData.budgets || 0 }}</div><div class="stat-label">Budgets</div></el-card></el-col>
        </el-row>
        <el-row :gutter="16" style="margin-top:12px">
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ dashData.overdueAr || 0 }}</div><div class="stat-label" style="color:#e6a23c">Overdue AR</div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ dashData.overdueAp || 0 }}</div><div class="stat-label" style="color:#e6a23c">Overdue AP</div></el-card></el-col>
        </el-row>
      </el-tab-pane>

      <!-- Subjects -->
      <el-tab-pane label="Account Subjects" name="subjects">
        <div class="section-toolbar">
          <el-button type="primary" @click="subjectDialog = true">Create Subject</el-button>
        </div>
        <el-table :data="subjects" border stripe v-loading="loading">
          <el-table-column prop="subjectCode" label="Code" width="110" />
          <el-table-column prop="subjectName" label="Name" min-width="160" />
          <el-table-column prop="subjectType" label="Type" width="100" />
          <el-table-column prop="level" label="Level" width="60" />
          <el-table-column prop="balanceDirection" label="Direction" width="80" />
          <el-table-column prop="status" label="Status" width="80">
            <template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column label="Actions" width="120">
            <template #default="{ row }">
              <el-button v-if="row.status === 'ACTIVE'" link type="warning" size="small" @click="toggleSubject(row.id, false)">Disable</el-button>
              <el-button v-else link type="success" size="small" @click="toggleSubject(row.id, true)">Enable</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Accounts -->
      <el-tab-pane label="Accounts" name="accounts">
        <div class="section-toolbar">
          <el-button type="primary" @click="accountDialog = true">Create Account</el-button>
        </div>
        <el-table :data="accounts" border stripe>
          <el-table-column prop="accountCode" label="Code" width="110" />
          <el-table-column prop="accountName" label="Name" min-width="160" />
          <el-table-column prop="accountType" label="Type" width="90" />
          <el-table-column prop="currency" label="Currency" width="70" />
          <el-table-column prop="balance" label="Balance" width="110">
            <template #default="{ row }">{{ row.balance?.toLocaleString() }}</template>
          </el-table-column>
          <el-table-column prop="status" label="Status" width="80">
            <template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column label="Actions" width="160">
            <template #default="{ row }">
              <el-button v-if="row.status === 'ACTIVE'" link type="warning" size="small" @click="freezeAccountAPI(row.id).then(loadFinanceAccounts)">Freeze</el-button>
              <el-button v-if="row.status === 'FROZEN'" link type="success" size="small" @click="unfreezeAccountAPI(row.id).then(loadFinanceAccounts)">Unfreeze</el-button>
              <el-button v-if="row.status !== 'CLOSED'" link type="danger" size="small" @click="closeAccountAPI(row.id).then(loadFinanceAccounts)">Close</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Vouchers -->
      <el-tab-pane label="Vouchers" name="vouchers">
        <div class="section-toolbar">
          <el-button type="primary" @click="voucherDialog = true">Create Voucher</el-button>
        </div>
        <el-table :data="vouchers" border stripe>
          <el-table-column prop="voucherNo" label="No." width="130" />
          <el-table-column prop="voucherType" label="Type" width="90" />
          <el-table-column prop="totalDebit" label="Total Debit" width="110">
            <template #default="{ row }">{{ row.totalDebit?.toLocaleString() }}</template>
          </el-table-column>
          <el-table-column prop="totalCredit" label="Total Credit" width="110">
            <template #default="{ row }">{{ row.totalCredit?.toLocaleString() }}</template>
          </el-table-column>
          <el-table-column prop="status" label="Status" width="100">
            <template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column label="Actions" width="280">
            <template #default="{ row }">
              <el-button v-if="row.status === 'DRAFT'" link type="primary" size="small" @click="submitV(row.id)">Submit</el-button>
              <el-button v-if="row.status === 'SUBMITTED'" link type="success" size="small" @click="approveV(row.id)">Approve</el-button>
              <el-button v-if="row.status === 'APPROVED'" link type="primary" size="small" @click="postV(row.id)">Post</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- AR/AP -->
      <el-tab-pane label="AR/AP" name="arap">
        <el-tabs>
          <el-tab-pane label="Accounts Receivable" name="ar">
            <div class="section-toolbar"><el-button type="primary" @click="arDialog = true">Create AR</el-button></div>
            <el-table :data="arRecords" border stripe>
              <el-table-column prop="documentCode" label="Doc Code" width="130" />
              <el-table-column prop="customerOrSupplierName" label="Customer" min-width="150" />
              <el-table-column prop="amount" label="Amount" width="110" />
              <el-table-column prop="balance" label="Balance" width="110" />
              <el-table-column prop="status" label="Status" width="90"><template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template></el-table-column>
              <el-table-column prop="dueDate" label="Due Date" width="110" />
            </el-table>
          </el-tab-pane>
          <el-tab-pane label="Accounts Payable" name="ap">
            <div class="section-toolbar"><el-button type="primary" @click="apDialog = true">Create AP</el-button></div>
            <el-table :data="apRecords" border stripe>
              <el-table-column prop="documentCode" label="Doc Code" width="130" />
              <el-table-column prop="customerOrSupplierName" label="Supplier" min-width="150" />
              <el-table-column prop="amount" label="Amount" width="110" />
              <el-table-column prop="balance" label="Balance" width="110" />
              <el-table-column prop="status" label="Status" width="90"><template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template></el-table-column>
              <el-table-column prop="dueDate" label="Due Date" width="110" />
            </el-table>
          </el-tab-pane>
        </el-tabs>
      </el-tab-pane>

      <!-- Cost Accounting -->
      <el-tab-pane label="Cost Accounting" name="cost">
        <el-tabs>
          <el-tab-pane label="Cost Centers" name="cc">
            <el-table :data="costCenters" border stripe>
              <el-table-column prop="code" label="Code" width="110" />
              <el-table-column prop="name" label="Name" min-width="180" />
              <el-table-column prop="type" label="Type" width="100" />
              <el-table-column prop="status" label="Status" width="80"><template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template></el-table-column>
            </el-table>
          </el-tab-pane>
          <el-tab-pane label="Standard Costs" name="sc">
            <el-table :data="standardCosts" border stripe>
              <el-table-column prop="id" label="ID" width="60" />
              <el-table-column prop="productCode" label="Product" width="120" />
              <el-table-column prop="materialCost" label="Material" width="110" />
              <el-table-column prop="laborCost" label="Labor" width="110" />
              <el-table-column prop="overheadCost" label="Overhead" width="110" />
              <el-table-column prop="totalCost" label="Total" width="110" />
            </el-table>
          </el-tab-pane>
          <el-tab-pane label="Cost Variances" name="cv">
            <el-table :data="costVariances" border stripe>
              <el-table-column prop="id" label="ID" width="60" />
              <el-table-column prop="productCode" label="Product" width="120" />
              <el-table-column prop="varianceType" label="Type" width="120" />
              <el-table-column prop="varianceAmount" label="Amount" width="110" />
              <el-table-column prop="variancePercentage" label="%" width="80" />
            </el-table>
          </el-tab-pane>
        </el-tabs>
      </el-tab-pane>

      <!-- Ratios -->
      <el-tab-pane label="Financial Ratios" name="ratios">
        <el-table :data="ratios" border stripe>
          <el-table-column prop="ratioCode" label="Code" width="120" />
          <el-table-column prop="ratioName" label="Name" min-width="200" />
          <el-table-column prop="value" label="Value" width="120" />
          <el-table-column prop="period" label="Period" width="100" />
          <el-table-column prop="benchmark" label="Benchmark" width="120" />
        </el-table>
      </el-tab-pane>
    </el-tabs>

    <!-- Dialogs -->
    <el-dialog v-model="subjectDialog" title="Create Subject" width="450px">
      <el-form :model="subjectForm" label-width="120px">
        <el-form-item label="Code"><el-input v-model="subjectForm.subjectCode" /></el-form-item>
        <el-form-item label="Name"><el-input v-model="subjectForm.subjectName" /></el-form-item>
        <el-form-item label="Type">
          <el-select v-model="subjectForm.subjectType" style="width:100%">
            <el-option label="Asset" value="ASSET" /><el-option label="Liability" value="LIABILITY" />
            <el-option label="Equity" value="EQUITY" /><el-option label="Revenue" value="REVENUE" />
            <el-option label="Expense" value="EXPENSE" />
          </el-select>
        </el-form-item>
        <el-form-item label="Direction">
          <el-select v-model="subjectForm.balanceDirection" style="width:100%">
            <el-option label="Debit" value="DEBIT" /><el-option label="Credit" value="CREDIT" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="subjectDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createSubject">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="accountDialog" title="Create Account" width="450px">
      <el-form :model="accountForm" label-width="120px">
        <el-form-item label="Code"><el-input v-model="accountForm.accountCode" /></el-form-item>
        <el-form-item label="Name"><el-input v-model="accountForm.accountName" /></el-form-item>
        <el-form-item label="Type">
          <el-select v-model="accountForm.accountType" style="width:100%">
            <el-option label="Cash" value="CASH" /><el-option label="Bank" value="BANK" />
            <el-option label="Receivable" value="RECEIVABLE" /><el-option label="Payable" value="PAYABLE" />
            <el-option label="Fixed Asset" value="FIXED_ASSET" />
          </el-select>
        </el-form-item>
        <el-form-item label="Currency">
          <el-select v-model="accountForm.currency" style="width:100%">
            <el-option label="CNY" value="CNY" /><el-option label="USD" value="USD" /><el-option label="EUR" value="EUR" />
          </el-select>
        </el-form-item>
        <el-form-item label="Description"><el-input v-model="accountForm.description" type="textarea" :rows="2" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="accountDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createFinanceAccount">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="voucherDialog" title="Create Voucher" width="450px">
      <el-form :model="voucherForm" label-width="120px">
        <el-form-item label="Voucher No"><el-input v-model="voucherForm.voucherNo" /></el-form-item>
        <el-form-item label="Type">
          <el-select v-model="voucherForm.voucherType" style="width:100%">
            <el-option label="General" value="GENERAL" /><el-option label="Payment" value="PAYMENT" />
            <el-option label="Receipt" value="RECEIPT" /><el-option label="Transfer" value="TRANSFER" />
          </el-select>
        </el-form-item>
        <el-form-item label="Remark"><el-input v-model="voucherForm.remark" type="textarea" :rows="2" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="voucherDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createVoucher">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="arDialog" title="Create AR" width="400px">
      <el-form :model="arForm" label-width="130px">
        <el-form-item label="Customer"><el-input v-model="arForm.customerOrSupplierName" /></el-form-item>
        <el-form-item label="Amount"><el-input-number v-model="arForm.amount" :min="0" style="width:100%" /></el-form-item>
        <el-form-item label="Due Date"><el-input v-model="arForm.dueDate" placeholder="YYYY-MM-DD" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="arDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createAr">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="apDialog" title="Create AP" width="400px">
      <el-form :model="apForm" label-width="130px">
        <el-form-item label="Supplier"><el-input v-model="apForm.customerOrSupplierName" /></el-form-item>
        <el-form-item label="Amount"><el-input-number v-model="apForm.amount" :min="0" style="width:100%" /></el-form-item>
        <el-form-item label="Due Date"><el-input v-model="apForm.dueDate" placeholder="YYYY-MM-DD" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="apDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createAp">Create</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.fi-container { padding: 20px; }
.toolbar { margin-bottom: 16px; }
.stat-value { font-size: 28px; font-weight: bold; color: #303133; text-align: center; }
.stat-label { font-size: 14px; color: #909399; text-align: center; margin-top: 5px; }
.section-toolbar { margin-bottom: 12px; }
</style>
