import http from '@/utils/http'

export interface ProcurementOrder {
  id?: number
  orderNo: string
  supplierCode: string
  supplierName: string
  warehouseId?: number
  warehouseName?: string
  purchaseType?: string
  status: string
  totalAmount?: number
  currency?: string
  paymentTerms?: string
  deliveryTerms?: string
  remark?: string
  approver?: string
  approvedAt?: string
  expectedDeliveryDate?: string
  actualDeliveryDate?: string
  createdAt?: string
}

export interface ProcurementQueryParam {
  pageNum?: number
  pageSize?: number
  status?: string
  orderNo?: string
  supplierCode?: string
}

export function getProcurementListAPI(params: ProcurementQueryParam) {
  return http<ProcurementOrder[]>({
    url: '/api/procurement/orders',
    method: 'get',
    params: params,
  })
}

export function getProcurementByOrderNoAPI(orderNo: string) {
  return http<ProcurementOrder>({
    url: '/api/procurement/orders/' + orderNo,
    method: 'get',
  })
}

export function createProcurementOrderAPI(data: ProcurementOrder) {
  return http<ProcurementOrder>({
    url: '/api/procurement/orders',
    method: 'post',
    data: data,
  })
}

export function approveProcurementOrderAPI(orderNo: string, approver: string) {
  return http<ProcurementOrder>({
    url: '/api/procurement/orders/' + orderNo + '/approve',
    method: 'post',
    params: { approver },
  })
}

export function rejectProcurementOrderAPI(orderNo: string, reason: string) {
  return http<ProcurementOrder>({
    url: '/api/procurement/orders/' + orderNo + '/reject',
    method: 'post',
    params: { reason },
  })
}