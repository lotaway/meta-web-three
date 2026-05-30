import http from '@/utils/http'
import type { CommonResult } from '@/types/common'

export interface Budget {
  id: number
  budgetCode: string
  budgetName: string
  type: string
  period: string
  departmentId: number
  departmentName: string
  status: string
  totalAmount: number
  usedAmount: number
  adjustedAmount: number
  currency: string
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  approvedAt: string | null
  approvedBy: number | null
  approverName: string | null
  remark: string | null
  lines: BudgetLine[]
}

export interface BudgetLine {
  id: number
  budgetId: number
  subjectCode: string
  subjectName: string
  budgetAmount: number
  usedAmount: number
  adjustedAmount: number
  sort: number
  remark: string | null
}

export interface BudgetAdjustment {
  id: number
  budgetId: number
  budgetCode: string
  adjustmentNo: string
  type: string
  status: string
  subjectCode: string
  subjectName: string
  originalAmount: number
  adjustedAmount: number
  afterAmount: number
  reason: string
  applicantId: number
  applicantName: string
  appliedAt: string
  approverId: number | null
  approverName: string | null
  approvedAt: string | null
  approvalComment: string | null
}

export interface BudgetCreateCommand {
  budgetCode: string
  budgetName: string
  type: string
  period: string
  departmentId: number
  departmentName: string
  createdBy: number
  creatorName: string
  remark: string
  lines: BudgetLineCreateCommand[]
}

export interface BudgetLineCreateCommand {
  subjectCode: string
  subjectName: string
  budgetAmount: number
  remark: string
}

export interface BudgetUpdateCommand {
  id: number
  budgetName: string
  departmentId: number
  departmentName: string
  remark: string
  lines: BudgetLineCreateCommand[]
}

export interface BudgetAdjustmentCommand {
  budgetId: number
  type: string
  subjectCode: string
  subjectName: string
  originalAmount: number
  adjustedAmount: number
  applicantId: number
  applicantName: string
  reason: string
}

export interface BudgetAnalysisResult {
  budgetId: number
  budgetCode: string
  budgetName: string
  totalBudget: number
  adjustedBudget: number
  usedAmount: number
  availableAmount: number
  usageRate: number
  lineAnalyses: BudgetLineAnalysis[]
}

export interface BudgetLineAnalysis {
  subjectCode: string
  subjectName: string
  budgetAmount: number
  adjustedAmount: number
  usedAmount: number
  availableAmount: number
  usageRate: number
}

export interface BudgetComparisonResult {
  budgetId: number
  budgetCode: string
  budgetAmount: number
  actualAmount: number
  variance: number
  varianceRate: number
  withinBudget: boolean
}

export function createBudgetAPI(data: BudgetCreateCommand) {
  return http<{ id: number }>({ url: '/api/budget', method: 'post', data })
}

export function updateBudgetAPI(id: number, data: BudgetUpdateCommand) {
  return http({ url: `/api/budget/${id}`, method: 'put', data: { ...data, id } })
}

export function getBudgetAPI(id: number) {
  return http<Budget>({ url: `/api/budget/${id}`, method: 'get' })
}

export function getBudgetByCodeAPI(code: string) {
  return http<Budget>({ url: `/api/budget/code/${code}`, method: 'get' })
}

export function listBudgetsAPI(params: { departmentId?: number; status?: string; period?: string }) {
  return http<Budget[]>({ url: '/api/budget/list', method: 'get', params })
}

export function getBudgetLinesAPI(budgetId: number) {
  return http<BudgetLine[]>({ url: `/api/budget/${budgetId}/lines`, method: 'get' })
}

export function submitBudgetAPI(id: number) {
  return http({ url: `/api/budget/${id}/submit`, method: 'post' })
}

export function approveBudgetAPI(id: number, approverId: number, approverName: string) {
  return http({ url: `/api/budget/${id}/approve`, method: 'post', data: { approverId, approverName } })
}

export function rejectBudgetAPI(id: number) {
  return http({ url: `/api/budget/${id}/reject`, method: 'post' })
}

export function closeBudgetAPI(id: number) {
  return http({ url: `/api/budget/${id}/close`, method: 'post' })
}

export function deleteBudgetAPI(id: number) {
  return http({ url: `/api/budget/${id}`, method: 'delete' })
}

export function applyAdjustmentAPI(data: BudgetAdjustmentCommand) {
  return http<{ id: number }>({ url: '/api/budget/adjustment', method: 'post', data })
}

export function getPendingAdjustmentsAPI() {
  return http<BudgetAdjustment[]>({ url: '/api/budget/adjustment/pending', method: 'get' })
}

export function getAdjustmentsAPI(budgetId: number) {
  return http<BudgetAdjustment[]>({ url: `/api/budget/${budgetId}/adjustments`, method: 'get' })
}

export function approveAdjustmentAPI(id: number, approverId: number, approverName: string, comment: string) {
  return http({ url: `/api/budget/adjustment/${id}/approve`, method: 'post', data: { approverId, approverName, comment } })
}

export function rejectAdjustmentAPI(id: number, approverId: number, approverName: string, comment: string) {
  return http({ url: `/api/budget/adjustment/${id}/reject`, method: 'post', data: { approverId, approverName, comment } })
}

export function analyzeBudgetAPI(id: number) {
  return http<BudgetAnalysisResult>({ url: `/api/budget/${id}/analysis`, method: 'get' })
}

export function compareBudgetAPI(id: number, actualAmount: number) {
  return http<BudgetComparisonResult>({ url: `/api/budget/${id}/compare`, method: 'get', params: { actualAmount } })
}

export function recordUsageAPI(budgetId: number, subjectCode: string, amount: number) {
  return http({ url: `/api/budget/${budgetId}/usage`, method: 'post', data: { subjectCode, amount } })
}