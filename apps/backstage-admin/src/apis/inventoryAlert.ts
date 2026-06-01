import http from '@/utils/http'

export interface InventoryAlert {
  id: number
  productId: number
  skuId?: number
  productName: string
  skuCode?: string
  warehouseId: number
  warehouseName: string
  currentStock: number
  threshold: number
  alertLevel: number
  alertLevelDesc: string
  alertStatus: number
  alertStatusDesc: string
  alertMessage: string
  alertTime: string
  resolvedTime?: string
  remark?: string
}

export function getAlertListAPI() {
  return http<InventoryAlert[]>({ url: '/api/inventory-alert/list', method: 'get' })
}

export function getAlertsByStatusAPI(status: number) {
  return http<InventoryAlert[]>({ url: `/api/inventory-alert/status/${status}`, method: 'get' })
}

export function getAlertsByLevelAPI(alertLevel: number) {
  return http<InventoryAlert[]>({ url: `/api/inventory-alert/level/${alertLevel}`, method: 'get' })
}

export function getAlertsByProductAPI(productId: number) {
  return http<InventoryAlert[]>({ url: `/api/inventory-alert/product/${productId}`, method: 'get' })
}

export function getHighPriorityAlertsAPI() {
  return http<InventoryAlert[]>({ url: '/api/inventory-alert/high-priority', method: 'get' })
}

export function resolveAlertAPI(id: number, remark?: string) {
  return http<InventoryAlert>({ url: `/api/inventory-alert/${id}/resolve`, method: 'post', params: { remark } })
}

export function ignoreAlertAPI(id: number, remark?: string) {
  return http<InventoryAlert>({ url: `/api/inventory-alert/${id}/ignore`, method: 'post', params: { remark } })
}

export function checkInventoryAPI(productId: number, threshold: number) {
  return http<void>({ url: '/api/inventory-alert/check', method: 'post', params: { productId, threshold } })
}

export function analyzeTurnoverAPI(productId: number) {
  return http<InventoryAlert>({ url: `/api/inventory-alert/turnover/${productId}`, method: 'get' })
}