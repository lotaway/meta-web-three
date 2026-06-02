import http from '@/utils/http'

export interface ProductInfo {
  productId: string
  productName: string
  batchNumber?: string
  productionDate?: string
  expirationDate?: string
  manufacturer?: string
  category?: string
  productionLocation?: string
  status: 'PRODUCED' | 'IN_TRANSIT' | 'DELIVERED' | 'SOLD' | 'EXPIRED'
  isActive?: boolean
  registeredAt?: string
}

export interface TraceRecord {
  id: number
  productId: string
  batchNumber: string
  traceType: 'PRODUCTION' | 'TRANSPORTATION' | 'DELIVERY' | 'SALE'
  status: 'ACTIVE' | 'COMPLETED' | 'VERIFIED'
  createdAt: string
  events?: TraceEvent[]
}

export interface TraceEvent {
  id: number
  traceId: number
  eventType: 'PRODUCTION' | 'TRANSPORTATION' | 'DELIVERY' | 'SALE'
  location?: string
  description?: string
  timestamp: string
  operator?: string
  txHash?: string
}

export interface TraceQueryParam {
  pageNum?: number
  pageSize?: number
  productId?: string
  status?: string
}

export interface ProductListQuery {
  status?: string
  pageNum?: number
  pageSize?: number
}

export interface ProductListResponse {
  list: ProductInfo[]
  total: number
  pageNum: number
  pageSize: number
}

export function registerProductAPI(data: {
  productId: string
  productName: string
  batchNumber: string
  productionDate?: string
  expirationDate?: string
  manufacturer?: string
}) {
  return http<void>({ url: '/api/traceability/product', method: 'post', data })
}

export function getProductInfoAPI(productId: string) {
  return http<ProductInfo>({ url: `/api/traceability/product/${productId}`, method: 'get' })
}

export function getProductListAPI(params: ProductListQuery) {
  return http<ProductListResponse>({ url: '/api/traceability/product/list', method: 'get', params })
}

export function getProductTraceIdsAPI(productId: string) {
  return http<number[]>({ url: `/api/traceability/product/${productId}/traces`, method: 'get' })
}

export function createTraceRecordAPI(data: { productId: string; batchNumber: string }) {
  return http<{ traceId: number }>({ url: '/api/traceability/trace', method: 'post', data })
}

export function getTraceRecordAPI(traceId: number) {
  return http<TraceRecord>({ url: `/api/traceability/trace/${traceId}`, method: 'get' })
}

export function getTraceEventsAPI(traceId: number) {
  return http<TraceEvent[]>({ url: `/api/traceability/trace/${traceId}/events`, method: 'get' })
}

export function addTraceEventAPI(traceId: number, data: {
  eventType: string
  location?: string
  description?: string
  operator?: string
}) {
  return http<void>({ url: `/api/traceability/trace/${traceId}/event`, method: 'post', data })
}

export function recordProductionAPI(traceId: number, data: { location: string; qualityInfo: string }) {
  return http<void>({ url: `/api/traceability/trace/${traceId}/production`, method: 'post', data })
}

export function recordTransportationAPI(traceId: number, data: {
  fromLocation: string
  toLocation: string
  carrierInfo: string
}) {
  return http<void>({ url: `/api/traceability/trace/${traceId}/transportation`, method: 'post', data })
}

export function recordDeliveryAPI(traceId: number, data: { location: string; receiverInfo: string }) {
  return http<void>({ url: `/api/traceability/trace/${traceId}/delivery`, method: 'post', data })
}

export function recordSaleAPI(traceId: number, data: {
  buyerAddress: string
  saleLocation: string
  price: number
}) {
  return http<void>({ url: `/api/traceability/trace/${traceId}/sale`, method: 'post', data })
}

export function verifyProductAPI(productId: string, batchNumber: string) {
  return http<{ verified: boolean }>({
    url: '/api/traceability/verify',
    method: 'get',
    params: { productId, batchNumber },
  })
}