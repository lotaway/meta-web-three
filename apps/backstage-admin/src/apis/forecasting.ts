import http from '@/utils/http'

export interface Forecast {
  id: number
  skuCode: string
  skuName: string
  warehouseId: number
  forecastDate: string
  predictedQuantity: number
  modelName?: string
  algorithm?: string
  status: 'PENDING' | 'CONFIRMED' | 'ADJUSTED' | 'ACTUAL_RECORDED'
  actualQuantity?: number
  accuracy?: number
  createdAt: string
  updatedAt?: string
}

export interface ForecastModel {
  id: number
  modelName: string
  modelType: string
  algorithm: string
  featureConfig?: string
  trainingDays?: number
  status: 'DRAFT' | 'TRAINING' | 'TRAINED' | 'DEPLOYED' | 'ARCHIVED'
  createdAt: string
  updatedAt?: string
}

export interface ForecastQueryParam {
  pageNum?: number
  pageSize?: number
  skuCode?: string
  warehouseId?: number
  status?: string
  startDate?: string
  endDate?: string
}

// Forecast APIs
export function createForecastAPI(data: {
  skuCode: string
  skuName: string
  warehouseId: number
  forecastDate: string
  quantity: number
  modelName?: string
}) {
  return http<{ forecastId: number }>({ url: '/api/forecasting/forecast', method: 'post', data })
}

export function createForecastWithAlgorithmAPI(data: {
  skuCode: string
  skuName: string
  warehouseId: number
  forecastDate: string
  algorithm?: string
  windowSize?: number
}) {
  return http<{ forecastId: number }>({ url: '/api/forecasting/forecast/algorithm', method: 'post', data })
}

export function getForecastByIdAPI(id: number) {
  return http<Forecast>({ url: `/api/forecasting/forecast/${id}`, method: 'get' })
}

export function getForecastBySkuAPI(skuCode: string) {
  return http<Forecast[]>({ url: `/api/forecasting/forecast/sku/${skuCode}`, method: 'get' })
}

export function getForecastByWarehouseAPI(warehouseId: number) {
  return http<Forecast[]>({ url: `/api/forecasting/forecast/warehouse/${warehouseId}`, method: 'get' })
}

export function getForecastHistoryAPI(skuCode: string, startDate: string, endDate: string) {
  return http<Forecast[]>({
    url: '/api/forecasting/forecast/history',
    method: 'get',
    params: { skuCode, startDate, endDate },
  })
}

export function confirmForecastAPI(id: number) {
  return http<void>({ url: `/api/forecasting/forecast/${id}/confirm`, method: 'post' })
}

export function adjustForecastAPI(id: number, newQuantity: number) {
  return http<void>({ url: `/api/forecasting/forecast/${id}/adjust`, method: 'post', data: { newQuantity } })
}

export function recordActualSalesAPI(id: number, actualQuantity: number) {
  return http<void>({
    url: `/api/forecasting/forecast/${id}/record-actual`,
    method: 'post',
    data: { actualQuantity },
  })
}

// Model APIs
export function createModelAPI(data: {
  modelName: string
  modelType: string
  algorithm: string
  featureConfig?: string
  trainingDays?: number
}) {
  return http<{ modelId: number }>({ url: '/api/forecasting/model', method: 'post', data })
}

export function getModelByIdAPI(id: number) {
  return http<ForecastModel>({ url: `/api/forecasting/model/${id}`, method: 'get' })
}

export function getAllModelsAPI() {
  return http<ForecastModel[]>({ url: '/api/forecasting/model', method: 'get' })
}

export function trainModelAPI(id: number) {
  return http<void>({ url: `/api/forecasting/model/${id}/train`, method: 'post' })
}

export function deployModelAPI(id: number) {
  return http<void>({ url: `/api/forecasting/model/${id}/deploy`, method: 'post' })
}

// Sample Data Generation
export function generateSampleSalesHistoryAPI(data: {
  skuCode: string
  warehouseId: number
  days?: number
  baseQuantity?: number
}) {
  return http<{ message: string; skuCode: string; warehouseId: number; recordCount: number }>({
    url: '/api/forecasting/sales-history/sample',
    method: 'post',
    data,
  })
}