import http from '@/utils/http'

export interface AfterSale {
  id?: number
  orderId: number
  orderNo: string
  userId: number
  productId?: number
  skuId?: number
  productName?: string
  productImage?: string
  quantity?: number
  refundAmount?: number
  afterSaleType: number
  afterSaleTypeDesc?: string
  afterSaleStatus: number
  afterSaleStatusDesc?: string
  applyReason?: string
  rejectReason?: string
  applyTime?: string
  processTime?: string
  completeTime?: string
  remark?: string
}

export interface AfterSaleQueryParam {
  pageNum?: number
  pageSize?: number
  status?: number
  type?: number
  orderNo?: string
  userId?: string
  startDate?: string
  endDate?: string
}

export interface AfterSaleStatistic {
  totalCount: number
  pendingCount: number
  processingCount: number
  approvedCount: number
  rejectedCount: number
  completedCount: number
  totalRefundAmount: number
  todayCount: number
  weekCount: number
  monthCount: number
}

export interface AfterSaleProcessParam {
  id: number
  status: number
  rejectReason?: string
  remark?: string
}

export function getAfterSaleListAPI(params: AfterSaleQueryParam) {
  return http<{ data: AfterSale[]; total: number; pageNum: number; pageSize: number }>({
    url: '/api/after-sale/list',
    method: 'get',
    params: params,
  })
}

export function getAfterSaleByIdAPI(id: number) {
  return http<AfterSale>({
    url: '/api/after-sale/' + id,
    method: 'get',
  })
}

export function getAfterSaleByOrderIdAPI(orderId: number) {
  return http<AfterSale[]>({
    url: '/api/after-sale/order/' + orderId,
    method: 'get',
  })
}

export function getAfterSaleStatisticsAPI() {
  return http<AfterSaleStatistic>({
    url: '/api/after-sale/statistics',
    method: 'get',
  })
}

export function processAfterSaleAPI(data: AfterSaleProcessParam) {
  return http<AfterSale>({
    url: '/api/after-sale/process',
    method: 'post',
    data: data,
  })
}

export function batchApproveAfterSaleAPI(ids: number[]) {
  return http<{ success: boolean; count: number }>({
    url: '/api/after-sale/batch-approve',
    method: 'post',
    data: ids,
  })
}

export function batchRejectAfterSaleAPI(ids: number[], reason: string) {
  return http<{ success: boolean; count: number }>({
    url: '/api/after-sale/batch-reject',
    method: 'post',
    params: { reason },
    data: ids,
  })
}