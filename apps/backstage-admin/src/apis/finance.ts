import http from '@/utils/http'

// Account Subject (/api/finance/subjects)
export interface AccountSubject {
  id: number; subjectCode: string; subjectName: string; subjectType: string;
  level: number; parentId: number; status: string; balanceDirection: string;
  createdAt: string; updatedAt: string
}
export function createSubjectAPI(data: Record<string, any>) { return http<number>({ url: '/api/finance/subjects', method: 'post', data }) }
export function getSubjectAPI(id: number) { return http<AccountSubject>({ url: `/api/finance/subjects/${id}`, method: 'get' }) }
export function listSubjectsAPI(params?: Record<string, any>) { return http<AccountSubject[]>({ url: '/api/finance/subjects', method: 'get', params }) }
export function getSubjectByCodeAPI(code: string) { return http<AccountSubject>({ url: `/api/finance/subjects/code/${code}`, method: 'get' }) }
export function disableSubjectAPI(id: number) { return http<void>({ url: `/api/finance/subjects/${id}/disable`, method: 'post' }) }
export function enableSubjectAPI(id: number) { return http<void>({ url: `/api/finance/subjects/${id}/enable`, method: 'post' }) }

// Account (/api/finance/accounts)
export interface Account {
  id: number; accountCode: string; accountName: string; accountType: string;
  currency: string; balance: number; status: string; description: string;
  createdAt: string; updatedAt: string
}
export function createAccountAPI(data: Record<string, any>) { return http<number>({ url: '/api/finance/accounts', method: 'post', data }) }
export function getAccountAPI(id: number) { return http<Account>({ url: `/api/finance/accounts/${id}`, method: 'get' }) }
export function listAccountsAPI(params?: Record<string, any>) { return http<Account[]>({ url: '/api/finance/accounts', method: 'get', params }) }
export function freezeAccountAPI(id: number) { return http<void>({ url: `/api/finance/accounts/${id}/freeze`, method: 'post' }) }
export function unfreezeAccountAPI(id: number) { return http<void>({ url: `/api/finance/accounts/${id}/unfreeze`, method: 'post' }) }
export function closeAccountAPI(id: number) { return http<void>({ url: `/api/finance/accounts/${id}/close`, method: 'post' }) }

// Voucher (/api/finance/vouchers)
export interface Voucher {
  id: number; voucherNo: string; voucherDate: string; voucherType: string;
  status: string; totalDebit: number; totalCredit: number; remark: string;
  createdBy: string; approvedBy: string; postedAt: string; createdAt: string
}
export function createVoucherAPI(data: Record<string, any>) { return http<number>({ url: '/api/finance/vouchers', method: 'post', data }) }
export function getVoucherAPI(id: number) { return http<Voucher>({ url: `/api/finance/vouchers/${id}`, method: 'get' }) }
export function listVouchersAPI(params?: Record<string, any>) { return http<Voucher[]>({ url: '/api/finance/vouchers', method: 'get', params }) }
export function submitVoucherAPI(id: number) { return http<void>({ url: `/api/finance/vouchers/${id}/submit`, method: 'post' }) }
export function approveVoucherAPI(id: number, approver: string) { return http<void>({ url: `/api/finance/vouchers/${id}/approve`, method: 'post', params: { approver } }) }
export function rejectVoucherAPI(id: number, approver: string, reason: string) { return http<void>({ url: `/api/finance/vouchers/${id}/reject`, method: 'post', params: { approver, reason } }) }
export function postVoucherAPI(id: number) { return http<void>({ url: `/api/finance/vouchers/${id}/post`, method: 'post' }) }

// Ledger (/api/finance/ledger)
export interface Ledger {
  id: number; ledgerNo: string; periodYear: number; periodMonth: number;
  status: string; totalDebit: number; totalCredit: number; createdAt: string
}
export function generateLedgerAPI(data: Record<string, any>) { return http<Ledger>({ url: '/api/finance/ledger/generate', method: 'post', data }) }
export function postLedgerAPI(id: number) { return http<Ledger>({ url: `/api/finance/ledger/${id}/post`, method: 'post' }) }
export function closeLedgerAPI(id: number) { return http<Ledger>({ url: `/api/finance/ledger/${id}/close`, method: 'post' }) }
export function getLedgerAPI(id: number) { return http<Ledger>({ url: `/api/finance/ledger/${id}`, method: 'get' }) }
export function getLedgerByPeriodAPI(year: number, month: number) { return http<Ledger>({ url: '/api/finance/ledger/period', method: 'get', params: { year, month } }) }
export function listLedgersByStatusAPI(status: string) { return http<Ledger[]>({ url: `/api/finance/ledger/status/${status}`, method: 'get' }) }
export function getSubjectBalanceAPI(subjectCode: string, year: number, month: number) { return http<Record<string, any>>({ url: `/api/finance/ledger/subject/${subjectCode}/balance`, method: 'get', params: { year, month } }) }

// AR/AP (/api/finance/arap)
export interface ArApRecord {
  id: number; documentCode: string; documentType: string; amount: number;
  balance: number; status: string; customerOrSupplierId: number;
  customerOrSupplierName: string; dueDate: string; createdAt: string
}
export function createArAPI(data: Record<string, any>) { return http<ArApRecord>({ url: '/api/finance/arap/ar', method: 'post', data }) }
export function receiveArAPI(data: Record<string, any>) { return http<ArApRecord>({ url: '/api/finance/arap/ar/receive', method: 'post', data }) }
export function getArAPI(id: number) { return http<ArApRecord>({ url: `/api/finance/arap/ar/${id}`, method: 'get' }) }
export function listArAPI() { return http<ArApRecord[]>({ url: '/api/finance/arap/ar/list', method: 'get' }) }
export function getOverdueArAPI() { return http<ArApRecord[]>({ url: '/api/finance/arap/ar/overdue', method: 'get' }) }
export function createApAPI(data: Record<string, any>) { return http<ArApRecord>({ url: '/api/finance/arap/ap', method: 'post', data }) }
export function payApAPI(data: Record<string, any>) { return http<ArApRecord>({ url: '/api/finance/arap/ap/pay', method: 'post', data }) }
export function getApAPI(id: number) { return http<ArApRecord>({ url: `/api/finance/arap/ap/${id}`, method: 'get' }) }
export function listApAPI() { return http<ArApRecord[]>({ url: '/api/finance/arap/ap/list', method: 'get' }) }
export function getOverdueApAPI() { return http<ArApRecord[]>({ url: '/api/finance/arap/ap/overdue', method: 'get' }) }

// Financial Ratio (/api/finance/financial-ratio)
export function getRatioDashboardAPI(period?: string) { return http<Record<string, any>>({ url: '/api/finance/financial-ratio/dashboard', method: 'get', params: { period } }) }
export function getRatioDetailsAPI(ratioType: string, period: string) { return http<Record<string, any>>({ url: '/api/finance/financial-ratio/details', method: 'get', params: { ratioType, period } }) }
export function listRatiosAPI() { return http<any[]>({ url: '/api/finance/financial-ratio/list', method: 'get' }) }
export function getCurrentRatiosAPI(period?: string) { return http<Record<string, any>>({ url: '/api/finance/financial-ratio/current', method: 'get', params: { period } }) }

// Cost Accounting (/api/finance/cost)
export interface CostCenter { id: number; code: string; name: string; type: string; status: string; createdAt: string }
export function createCostCenterAPI(data: Record<string, any>) { return http<Record<string, any>>({ url: '/api/finance/cost/cost-center', method: 'post', data }) }
export function getCostCenterAPI(id: number) { return http<CostCenter>({ url: `/api/finance/cost/cost-center/${id}`, method: 'get' }) }
export function listCostCentersAPI(type?: string) { return http<CostCenter[]>({ url: '/api/finance/cost/cost-center/list', method: 'get', params: { type } }) }
export function createStandardCostAPI(data: Record<string, any>) { return http<Record<string, any>>({ url: '/api/finance/cost/standard-cost', method: 'post', data }) }
export function listStandardCostsAPI() { return http<any[]>({ url: '/api/finance/cost/standard-cost/list', method: 'get' }) }
export function createActualCostAPI(data: Record<string, any>) { return http<Record<string, any>>({ url: '/api/finance/cost/actual-cost', method: 'post', data }) }
export function listActualCostsAPI() { return http<any[]>({ url: '/api/finance/cost/actual-cost/list', method: 'get' }) }
export function listCostVariancesAPI() { return http<any[]>({ url: '/api/finance/cost/variance/list', method: 'get' }) }
