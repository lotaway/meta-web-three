import http from '@/utils/http'

// Settlement Types
export interface SettlementOrder {
  id: number
  settlementNo: string
  orderNo: string
  merchantId: number
  merchantName: string
  orderAmount: number
  settlementAmount: number
  commissionAmount: number
  refundAmount: number
  status: string
  channel: string
  settlementDate: string | null
  createdAt: string
  updatedAt: string
  description: string | null
}

export interface SettlementQueryParam {
  pageNum: number
  pageSize: number
  status?: string
  merchantId?: number
  startDate?: string
  endDate?: string
}

export interface SettlementRequest {
  settlementNo: string
  orderNo: string
  merchantId: number
  merchantName: string
  orderAmount: number
  commissionRate: number
}

// Settlement Status Enum
export const SettlementStatus = {
  PENDING: 'PENDING',
  CONFIRMED: 'CONFIRMED',
  PROCESSING: 'PROCESSING',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
  CANCELLED: 'CANCELLED'
} as const

export type SettlementStatusType = typeof SettlementStatus[keyof typeof SettlementStatus]

// Status Text Mapping
export const SettlementStatusText: Record<string, string> = {
  PENDING: 'settlement.status.pending',
  CONFIRMED: 'settlement.status.confirmed',
  PROCESSING: 'settlement.status.processing',
  COMPLETED: 'settlement.status.completed',
  FAILED: 'settlement.status.failed',
  CANCELLED: 'settlement.status.cancelled'
}

// API Functions
export function getSettlementListAPI(params: SettlementQueryParam) {
  return http<{ list: SettlementOrder[]; total: number }>({
    url: '/api/settlement',
    method: 'get',
    params
  })
}

export function getSettlementDetailAPI(id: number) {
  return http<SettlementOrder>({
    url: `/api/settlement/${id}`,
    method: 'get'
  })
}

export function createSettlementAPI(data: SettlementRequest) {
  return http<{ id: number }>({
    url: '/api/settlement',
    method: 'post',
    data
  })
}

export function confirmSettlementAPI(id: number) {
  return http({
    url: `/api/settlement/${id}/confirm`,
    method: 'post'
  })
}

export function processSettlementAPI(id: number) {
  return http({
    url: `/api/settlement/${id}/process`,
    method: 'post'
  })
}

export function completeSettlementAPI(id: number) {
  return http({
    url: `/api/settlement/${id}/complete`,
    method: 'post'
  })
}

export function failSettlementAPI(id: number, reason: string) {
  return http({
    url: `/api/settlement/${id}/fail`,
    method: 'post',
    params: { reason }
  })
}

export function cancelSettlementAPI(id: number) {
  return http({
    url: `/api/settlement/${id}/cancel`,
    method: 'post'
  })
}

export function refundSettlementAPI(id: number, amount: number) {
  return http({
    url: `/api/settlement/${id}/refund`,
    method: 'post',
    params: { amount }
  })
}

// Settlement Statistics
export interface SettlementStatistics {
  totalCount: number
  pendingCount: number
  processingCount: number
  completedCount: number
  totalAmount: number
  completedAmount: number
}

export function getSettlementStatisticsAPI() {
  return http<SettlementStatistics>({
    url: '/api/settlement/statistics',
    method: 'get'
  })
}