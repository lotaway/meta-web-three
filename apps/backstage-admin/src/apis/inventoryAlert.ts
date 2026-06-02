import http from '@/utils/http'

export interface InventoryAlert {
  id: number
  productId: number
  skuId?: number
  productName?: string
  skuCode?: string
  warehouseId?: number
  warehouseName?: string
  currentStock: number
  threshold: number
  alertLevel: number
  alertLevelDesc?: string
  alertStatus: number
  alertStatusDesc?: string
  alertMessage?: string
  alertTime?: string
  resolvedTime?: string
  remark?: string
}

export interface InventoryAlertQueryParam {
  pageNum?: number
  pageSize?: number
  productId?: number
  productName?: string
  alertStatus?: number
  alertLevel?: number
  startDate?: string
  endDate?: string
}

export interface InventoryAlertStatistics {
  total: number
  pending: number
  highPriority: number
  resolved: number
  ignored: number
}

export function getInventoryAlertListAPI(params: InventoryAlertQueryParam) {
  return http<{ list: InventoryAlert[]; total: number; pageNum: number; pageSize: number }>({
    url: '/api/admin/inventory-alert/list',
    method: 'get',
    params,
  })
}

export function getInventoryAlertByIdAPI(id: number) {
  return http<InventoryAlert>({ url: `/api/admin/inventory-alert/${id}`, method: 'get' })
}

export function resolveInventoryAlertAPI(id: number, remark?: string) {
  return http<InventoryAlert>({
    url: '/api/admin/inventory-alert/resolve',
    method: 'post',
    params: { id, remark },
  })
}

export function ignoreInventoryAlertAPI(id: number, remark?: string) {
  return http<InventoryAlert>({
    url: '/api/admin/inventory-alert/ignore',
    method: 'post',
    params: { id, remark },
  })
}

export function getInventoryAlertStatisticsAPI() {
  return http<InventoryAlertStatistics>({ url: '/api/admin/inventory-alert/statistics', method: 'get' })
}

export function getHighPriorityAlertsAPI() {
  return http<InventoryAlert[]>({ url: '/api/admin/inventory-alert/high-priority', method: 'get' })
}