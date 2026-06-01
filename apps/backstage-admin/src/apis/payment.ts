import http from '@/utils/http'
import type { CommonPage } from '@/types/common'

export interface PaymentOrder {
  id: number
  orderNo: string
  userId: number
  amount: number
  currency: string
  paymentMethod: string
  status: number
  transactionId: string
  createdAt: string
  updatedAt: string
}

export interface PaymentQueryParam {
  pageNum: number
  pageSize: number
  orderNo?: string
  userId?: number
  status?: number
  paymentMethod?: string
  startDate?: string
  endDate?: string
}

export interface ReconciliationReport {
  date: string
  totalOrders: number
  totalAmount: number
  successCount: number
  failedCount: number
  pendingCount: number
  differenceCount: number
}

export interface RefundRecord {
  id: number
  orderNo: string
  refundAmount: number
  refundReason: string
  status: number
  createdAt: string
  processedAt?: string
}

export const getPaymentOrderListAPI = (params: PaymentQueryParam) => {
  return http<CommonPage<PaymentOrder>>({
    url: '/exchange/orders',
    method: 'get',
    params: params,
  })
}

export const getPaymentOrderByIdAPI = (id: number) => {
  return http<PaymentOrder>({
    url: `/exchange/orders/${id}`,
    method: 'get',
  })
}

export const getReconciliationReportAPI = (date: string) => {
  return http<ReconciliationReport>({
    url: '/reconciliation/report',
    method: 'get',
    params: { date },
  })
}

export const getTodayReconciliationStatusAPI = () => {
  return http<ReconciliationReport>({
    url: '/reconciliation/report/today',
    method: 'get',
  })
}

export const getPendingDifferenceCountAPI = (date: string) => {
  return http<{ count: number }>({
    url: '/reconciliation/pending/count',
    method: 'get',
    params: { date },
  })
}

export const getRefundListAPI = (params: PaymentQueryParam) => {
  return http<CommonPage<RefundRecord>>({
    url: '/returnApply/list',
    method: 'get',
    params: params,
  })
}

export const updateRefundStatusAPI = (id: number, status: number) => {
  return http({
    url: `/returnApply/update/status/${id}`,
    method: 'post',
    data: { status },
  })
}

export const verifyPaymentAPI = (orderId: string, transactionId: string) => {
  return http({
    url: '/pay/verify',
    method: 'post',
    data: { orderId, transactionId },
  })
}

export const queryPaymentStatusAPI = (outTradeNo: string) => {
  return http({
    url: '/pay/alipay/query',
    method: 'get',
    params: { outTradeNo },
  })
}