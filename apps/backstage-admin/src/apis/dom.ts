import http from '@/utils/http'

export interface DomOrder {
  id?: number
  domOrderNo: string
  originalOrderNo: string
  customerId: string
  customerName: string
  status: string
  totalAmount?: number
  currency?: string
  sourcingStrategy?: string
  region?: string
  createdAt?: string
  updatedAt?: string
  lines?: DomOrderLine[]
  fulfillmentPlan?: FulfillmentPlan
}

export interface DomOrderLine {
  id?: number
  domOrderId?: number
  skuCode: string
  skuName: string
  quantity: number
  fulfilledQuantity?: number
  warehouseId?: number
  warehouseName?: string
  unitPrice?: number
  status?: string
}

export interface FulfillmentPlan {
  id?: number
  domOrderId?: number
  domOrderNo?: string
  totalLines?: number
  fulfilledLines?: number
  partiallyFulfilledLines?: number
  unfulfilledLines?: number
  status?: string
}

export interface DomQueryParam {
  pageNum?: number
  pageSize?: number
  status?: string
  domOrderNo?: string
}

export function getDomListAPI(params: DomQueryParam) {
  return http<DomOrder[]>({
    url: '/api/dom',
    method: 'get',
    params,
  })
}

export function getDomByIdAPI(id: number) {
  return http<DomOrder>({
    url: '/api/dom/' + id,
    method: 'get',
  })
}

export function getDomByNoAPI(domOrderNo: string) {
  return http<DomOrder>({
    url: '/api/dom/no/' + domOrderNo,
    method: 'get',
  })
}

export function createDomOrderAPI(data: {
  originalOrderNo: string
  customerId: string
  customerName: string
  region?: string
  sourcingStrategy?: string
  items: { skuCode: string; skuName: string; quantity: number; unitPrice: number }[]
}) {
  return http<DomOrder>({
    url: '/api/dom',
    method: 'post',
    data,
  })
}

export function checkDomAvailabilityAPI(id: number) {
  return http<void>({
    url: '/api/dom/' + id + '/check-atp',
    method: 'post',
  })
}

export function sourceDomOrderAPI(id: number) {
  return http<void>({
    url: '/api/dom/' + id + '/source',
    method: 'post',
  })
}

export function approveDomFulfillmentAPI(id: number) {
  return http<void>({
    url: '/api/dom/' + id + '/approve',
    method: 'post',
  })
}

export function cancelDomOrderAPI(id: number) {
  return http<void>({
    url: '/api/dom/' + id + '/cancel',
    method: 'post',
  })
}
