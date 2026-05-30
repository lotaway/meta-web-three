import http from '@/utils/http'

export interface AbcClassification {
  skuCode: string
  category: 'A' | 'B' | 'C'
  totalValue: number
  turnoverRate: number
  rank: number
}

export function classifyInventoryAPI(params?: { warehouseId?: number; periodDays?: number }) {
  return http<AbcClassification[]>({ url: '/api/inventory/abc/classify', method: 'get', params })
}

export interface DemandForecast {
  id: number
  skuCode: string
  warehouseId: number
  forecastPeriodDays: number
  predictedQuantity: number
  confidenceLevel: number
  forecastMethod: string
  forecastStartDate: string
  forecastEndDate: string
  status: 'PENDING' | 'APPROVED' | 'REJECTED'
  generatedAt: string
  notes?: string
}

export function generateForecastAPI(params: {
  skuCode: string
  warehouseId: number
  forecastDays?: number
  method?: string
}) {
  return http<DemandForecast>({ url: '/api/inventory/forecast/generate', method: 'post', params })
}

export function getPendingForecastsAPI() {
  return http<DemandForecast[]>({ url: '/api/inventory/forecast/pending', method: 'get' })
}

export function approveForecastAPI(id: number) {
  return http<DemandForecast>({ url: `/api/inventory/forecast/${id}/approve`, method: 'post' })
}

export function rejectForecastAPI(id: number) {
  return http<DemandForecast>({ url: `/api/inventory/forecast/${id}/reject`, method: 'post' })
}

export function queryForecastByIdAPI(id: number) {
  return http<DemandForecast>({ url: `/api/inventory/forecast/${id}`, method: 'get' })
}