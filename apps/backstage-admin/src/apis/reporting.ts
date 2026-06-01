import http from '@/utils/http'
import type { CommonResult } from '@/types/common'

// Report Types
export interface BalanceSheetReport {
  asOfDate: string
  totalAssets: number
  totalLiabilities: number
  totalEquity: number
  assets: AssetItem[]
  liabilities: LiabilityItem[]
  equity: EquityItem[]
}

export interface AssetItem {
  code: string
  name: string
  currentAmount: number
  previousAmount: number
  change: number
  changePercent: number
}

export interface LiabilityItem {
  code: string
  name: string
  currentAmount: number
  previousAmount: number
  change: number
  changePercent: number
}

export interface EquityItem {
  code: string
  name: string
  currentAmount: number
  previousAmount: number
  change: number
  changePercent: number
}

export interface IncomeStatementReport {
  startDate: string
  endDate: string
  totalRevenue: number
  totalCost: number
  grossProfit: number
  totalExpenses: number
  operatingProfit: number
  netProfit: number
  revenue: RevenueItem[]
  cost: CostItem[]
  expenses: ExpenseItem[]
}

export interface RevenueItem {
  code: string
  name: string
  amount: number
  proportion: number
}

export interface CostItem {
  code: string
  name: string
  amount: number
  proportion: number
}

export interface ExpenseItem {
  code: string
  name: string
  amount: number
  proportion: number
}

export interface TrialBalanceReport {
  asOfDate: string
  totalDebit: number
  totalCredit: number
  isBalanced: boolean
  accounts: TrialBalanceItem[]
}

export interface TrialBalanceItem {
  subjectCode: string
  subjectName: string
  debitBalance: number
  creditBalance: number
  debitCredit: string
}

// Balance Sheet APIs
export const getBalanceSheet = (asOfDate: string) => {
  return http<CommonResult<BalanceSheetReport>>({
    url: '/api/finance/reports/balance-sheet',
    method: 'get',
    params: { asOfDate }
  })
}

// Income Statement APIs
export const getIncomeStatement = (startDate: string, endDate: string) => {
  return http<CommonResult<IncomeStatementReport>>({
    url: '/api/finance/reports/income-statement',
    method: 'get',
    params: { startDate, endDate }
  })
}

// Trial Balance APIs
export const getTrialBalance = (asOfDate: string) => {
  return http<CommonResult<TrialBalanceReport>>({
    url: '/api/finance/reports/trial-balance',
    method: 'get',
    params: { asOfDate }
  })
}

// Export APIs
export const exportBalanceSheet = (asOfDate: string) => {
  return http<Blob>({
    url: '/api/finance/reports/balance-sheet/export',
    method: 'get',
    params: { asOfDate },
    responseType: 'blob'
  })
}

export const exportIncomeStatement = (startDate: string, endDate: string) => {
  return http<Blob>({
    url: '/api/finance/reports/income-statement/export',
    method: 'get',
    params: { startDate, endDate },
    responseType: 'blob'
  })
}

export const exportTrialBalance = (asOfDate: string) => {
  return http<Blob>({
    url: '/api/finance/reports/trial-balance/export',
    method: 'get',
    params: { asOfDate },
    responseType: 'blob'
  })
}