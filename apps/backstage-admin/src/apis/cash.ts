import http from '@/utils/http'

// Cash Plan Types
export interface CashPlan {
  id: number
  planCode: string
  planName: string
  type: string
  period: string
  startDate: string
  endDate: string
  status: string
  totalAmount: number
  inflowAmount: number
  outflowAmount: number
  currency: string
  departmentId: number
  departmentName: string
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  approvedAt: string | null
  approvedBy: number | null
  approverName: string | null
  remark: string | null
  lines: CashPlanLine[]
}

export interface CashPlanLine {
  id: number
  cashPlanId: number
  categoryCode: string
  categoryName: string
  flowDirection: string
  plannedAmount: number
  actualAmount: number
  plannedDate: string
  remark: string | null
  sort: number
}

// Bank Account Types
export interface BankAccount {
  id: number
  accountCode: string
  accountName: string
  bankName: string
  accountNumber: string
  accountType: string
  status: string
  balance: number
  frozenAmount: number
  currency: string
  remark: string | null
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  isActive: boolean
}

// Cash Transfer Types
export interface CashTransfer {
  id: number
  transferNo: string
  fromAccountId: number
  fromAccountName: string
  toAccountId: number
  toAccountName: string
  amount: number
  currency: string
  status: string
  type: string
  purpose: string
  transferDate: string
  remark: string | null
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  approvedAt: string | null
  approvedBy: number | null
  approverName: string | null
  executedAt: string | null
  executorName: string | null
}

// Bank Reconciliation Types
export interface BankReconciliation {
  id: number
  reconciliationNo: string
  bankAccountId: number
  bankAccountName: string
  bankName: string
  statementDate: string
  statementEndDate: string
  bankBalance: number
  bookBalance: number
  variance: number
  status: string
  remark: string | null
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  approvedAt: string | null
  approvedBy: number | null
  approverName: string | null
}

// Cash Flow Forecast Types
export interface CashFlowForecast {
  id: number
  forecastNo: string
  forecastDate: string
  startDate: string
  endDate: string
  currency: string
  openingBalance: number
  predictedInflow: number
  predictedOutflow: number
  predictedClosingBalance: number
  remark: string | null
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  items: ForecastItem[]
}

export interface ForecastItem {
  id: number
  forecastId: number
  categoryCode: string
  categoryName: string
  flowDirection: string
  amount: number
  predictedDate: string
  description: string | null
  confidenceLevel: number
  remark: string | null
}

// Cash Summary
export interface CashSummary {
  totalBalance: number
  activeAccountCount: number
  pendingTransferCount: number
  pendingReconciliationCount: number
  draftPlanCount: number
}

// Cash Plan API
export function createCashPlanAPI(data: any) {
  return http<{ id: number }>({ url: '/api/cash/plan', method: 'post', data })
}

export function updateCashPlanAPI(id: number, data: any) {
  return http({ url: `/api/cash/plan/${id}`, method: 'put', data: { ...data, id } })
}

export function getCashPlanAPI(id: number) {
  return http<CashPlan>({ url: `/api/cash/plan/${id}`, method: 'get' })
}

export function getCashPlanByCodeAPI(code: string) {
  return http<CashPlan>({ url: `/api/cash/plan/code/${code}`, method: 'get' })
}

export function listCashPlansAPI(params?: { departmentId?: number; status?: string }) {
  return http<CashPlan[]>({ url: '/api/cash/plan/list', method: 'get', params })
}

export function getCashPlanLinesAPI(id: number) {
  return http<CashPlanLine[]>({ url: `/api/cash/plan/${id}/lines`, method: 'get' })
}

export function submitCashPlanAPI(id: number) {
  return http({ url: `/api/cash/plan/${id}/submit`, method: 'post' })
}

export function approveCashPlanAPI(id: number, approverId: number, approverName: string) {
  return http({ url: `/api/cash/plan/${id}/approve`, method: 'post', data: { approverId, approverName } })
}

export function rejectCashPlanAPI(id: number) {
  return http({ url: `/api/cash/plan/${id}/reject`, method: 'post' })
}

export function deleteCashPlanAPI(id: number) {
  return http({ url: `/api/cash/plan/${id}`, method: 'delete' })
}

// Bank Account API
export function createBankAccountAPI(data: any) {
  return http<{ id: number }>({ url: '/api/cash/account', method: 'post', data })
}

export function getBankAccountAPI(id: number) {
  return http<BankAccount>({ url: `/api/cash/account/${id}`, method: 'get' })
}

export function getBankAccountByCodeAPI(code: string) {
  return http<BankAccount>({ url: `/api/cash/account/code/${code}`, method: 'get' })
}

export function listBankAccountsAPI(params?: { status?: string }) {
  return http<BankAccount[]>({ url: '/api/cash/account/list', method: 'get', params })
}

export function listActiveBankAccountsAPI() {
  return http<BankAccount[]>({ url: '/api/cash/account/active', method: 'get' })
}

export function getTotalCashBalanceAPI() {
  return http<{ totalBalance: number }>({ url: '/api/cash/account/total-balance', method: 'get' })
}

export function freezeBankAccountAPI(id: number) {
  return http({ url: `/api/cash/account/${id}/freeze`, method: 'post' })
}

export function unfreezeBankAccountAPI(id: number) {
  return http({ url: `/api/cash/account/${id}/unfreeze`, method: 'post' })
}

export function closeBankAccountAPI(id: number) {
  return http({ url: `/api/cash/account/${id}/close`, method: 'post' })
}

export function deleteBankAccountAPI(id: number) {
  return http({ url: `/api/cash/account/${id}`, method: 'delete' })
}

// Cash Transfer API
export function createCashTransferAPI(data: any) {
  return http<{ id: number }>({ url: '/api/cash/transfer', method: 'post', data })
}

export function getCashTransferAPI(id: number) {
  return http<CashTransfer>({ url: `/api/cash/transfer/${id}`, method: 'get' })
}

export function listCashTransfersAPI(params?: { status?: string; accountId?: number }) {
  return http<CashTransfer[]>({ url: '/api/cash/transfer/list', method: 'get', params })
}

export function submitCashTransferAPI(id: number) {
  return http({ url: `/api/cash/transfer/${id}/submit`, method: 'post' })
}

export function approveCashTransferAPI(id: number, approverId: number, approverName: string) {
  return http({ url: `/api/cash/transfer/${id}/approve`, method: 'post', data: { approverId, approverName } })
}

export function rejectCashTransferAPI(id: number) {
  return http({ url: `/api/cash/transfer/${id}/reject`, method: 'post' })
}

export function cancelCashTransferAPI(id: number) {
  return http({ url: `/api/cash/transfer/${id}/cancel`, method: 'post' })
}

export function deleteCashTransferAPI(id: number) {
  return http({ url: `/api/cash/transfer/${id}`, method: 'delete' })
}

// Bank Reconciliation API
export function createBankReconciliationAPI(data: any) {
  return http<{ id: number }>({ url: '/api/cash/reconciliation', method: 'post', data })
}

export function getBankReconciliationAPI(id: number) {
  return http<BankReconciliation>({ url: `/api/cash/reconciliation/${id}`, method: 'get' })
}

export function listBankReconciliationsAPI(params?: { bankAccountId?: number; status?: string }) {
  return http<BankReconciliation[]>({ url: '/api/cash/reconciliation/list', method: 'get', params })
}

export function submitBankReconciliationAPI(id: number) {
  return http({ url: `/api/cash/reconciliation/${id}/submit`, method: 'post' })
}

export function approveBankReconciliationAPI(id: number, approverId: number, approverName: string) {
  return http({ url: `/api/cash/reconciliation/${id}/approve`, method: 'post', data: { approverId, approverName } })
}

export function deleteBankReconciliationAPI(id: number) {
  return http({ url: `/api/cash/reconciliation/${id}`, method: 'delete' })
}

// Cash Flow Forecast API
export function createCashFlowForecastAPI(data: any) {
  return http<{ id: number }>({ url: '/api/cash/forecast', method: 'post', data })
}

export function getCashFlowForecastAPI(id: number) {
  return http<CashFlowForecast>({ url: `/api/cash/forecast/${id}`, method: 'get' })
}

export function listCashFlowForecastsAPI(params?: { forecastDate?: string }) {
  return http<CashFlowForecast[]>({ url: '/api/cash/forecast/list', method: 'get', params })
}

export function getCashFlowForecastItemsAPI(id: number) {
  return http<ForecastItem[]>({ url: `/api/cash/forecast/${id}/items`, method: 'get' })
}

export function deleteCashFlowForecastAPI(id: number) {
  return http({ url: `/api/cash/forecast/${id}`, method: 'delete' })
}

// Dashboard API
export function getCashSummaryAPI() {
  return http<CashSummary>({ url: '/api/cash/summary', method: 'get' })
}