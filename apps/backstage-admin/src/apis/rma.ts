import http from '@/utils/http'

export interface RmaOrder {
  id?: number
  rmaNo: string
  orderNo: string
  returnType: string
  status: string
  customerId: string
  customerName: string
  contactPhone?: string
  reasonCode: string
  reasonDescription?: string
  warehouseId?: number
  totalQuantity?: number
  totalAmount?: number
  currency?: string
  createdBy?: string
  createdAt?: string
  updatedAt?: string
}

export interface RmaOrderItem {
  id?: number
  rmaId?: number
  skuCode: string
  skuName: string
  expectedQuantity: number
  inspectedQuantity?: number
  acceptedQuantity?: number
  unitPrice?: number
  reasonCode?: string
  reasonDescription?: string
}

export interface RmaQueryParam {
  pageNum?: number
  pageSize?: number
  status?: string
  rmaNo?: string
  orderNo?: string
}

export interface RmaInspection {
  id?: number
  rmaId: number
  rmaNo?: string
  inspector: string
  inspectionDate?: string
  result: string
  conclusion?: string
  totalInspected?: number
  totalPassed?: number
  totalFailed?: number
  remark?: string
}

export interface RmaDisposition {
  id?: number
  rmaId: number
  rmaNo?: string
  dispositionType: string
  refundAmount?: number
  replacementSkuCode?: string
  replacementQuantity?: number
  scrapQuantity?: number
  scrapReason?: string
  remark?: string
}

export function getRmaListAPI(params: RmaQueryParam) {
  return http<RmaOrder[]>({
    url: '/api/rma',
    method: 'get',
    params,
  })
}

export function getRmaByIdAPI(id: number) {
  return http<RmaOrder>({
    url: '/api/rma/' + id,
    method: 'get',
  })
}

export function getRmaByNoAPI(rmaNo: string) {
  return http<RmaOrder>({
    url: '/api/rma/no/' + rmaNo,
    method: 'get',
  })
}

export function createRmaAPI(data: {
  orderNo: string
  returnType: string
  customerId: string
  customerName: string
  contactPhone?: string
  reasonCode: string
  reasonDescription?: string
  warehouseId?: number
  items: RmaOrderItem[]
}) {
  return http<RmaOrder>({
    url: '/api/rma',
    method: 'post',
    data,
  })
}

export function submitRmaForInspectionAPI(id: number) {
  return http<void>({
    url: '/api/rma/' + id + '/submit-inspection',
    method: 'post',
  })
}

export function recordRmaInspectionAPI(id: number, data: RmaInspection) {
  return http<void>({
    url: '/api/rma/' + id + '/record-inspection',
    method: 'post',
    data,
  })
}

export function makeRmaDispositionAPI(id: number, data: RmaDisposition) {
  return http<void>({
    url: '/api/rma/' + id + '/disposition',
    method: 'post',
    data,
  })
}

export function executeRmaDispositionAPI(id: number) {
  return http<void>({
    url: '/api/rma/' + id + '/execute',
    method: 'post',
  })
}

export function cancelRmaAPI(id: number) {
  return http<void>({
    url: '/api/rma/' + id + '/cancel',
    method: 'post',
  })
}
