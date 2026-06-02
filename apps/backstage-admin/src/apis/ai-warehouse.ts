import http from '@/utils/http'

export interface AIWarehouseRequest {
  id?: number
  warehouseId: number
  warehouseName: string
  capabilityType: string
  requestData: string
  responseData?: string
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED'
  createdAt?: string
  updatedAt?: string
}

export interface AICapability {
  id: number
  name: string
  type: string
  description: string
  enabled: boolean
  config?: Record<string, any>
}

export interface WarehouseQueryParam {
  pageNum?: number
  pageSize?: number
  warehouseId?: number
  capabilityType?: string
  status?: string
}

// Request APIs
export function listAIWarehouseRequestsAPI(params: WarehouseQueryParam) {
  return http<{ list: AIWarehouseRequest[]; total: number }>({
    url: '/api/ai-warehouse/requests',
    method: 'get',
    params
  })
}

export function getAIWarehouseRequestAPI(id: number) {
  return http<AIWarehouseRequest>({
    url: `/api/ai-warehouse/requests/${id}`,
    method: 'get'
  })
}

export function createAIWarehouseRequestAPI(data: Partial<AIWarehouseRequest>) {
  return http<{ id: number }>({
    url: '/api/ai-warehouse/requests',
    method: 'post',
    data
  })
}

export function updateAIWarehouseRequestAPI(id: number, data: Partial<AIWarehouseRequest>) {
  return http<void>({
    url: `/api/ai-warehouse/requests/${id}`,
    method: 'put',
    data
  })
}

export function deleteAIWarehouseRequestAPI(id: number) {
  return http<void>({
    url: `/api/ai-warehouse/requests/${id}`,
    method: 'delete'
  })
}

// Capability APIs
export function listAICapabilitiesAPI() {
  return http<AICapability[]>({
    url: '/api/ai-warehouse/capabilities',
    method: 'get'
  })
}

export function updateAICapabilityAPI(id: number, data: Partial<AICapability>) {
  return http<void>({
    url: `/api/ai-warehouse/capabilities/${id}`,
    method: 'put',
    data
  })
}

export function enableAICapabilityAPI(id: number, enabled: boolean) {
  return http<void>({
    url: `/api/ai-warehouse/capabilities/${id}/enable`,
    method: 'put',
    data: { enabled }
  })
}

// AI Feature APIs
export function getLocationRecommendationAPI(warehouseId: number, params: Record<string, any>) {
  return http<{ recommendations: Array<{ locationId: number; score: number; reason: string }> }>({
    url: `/api/ai-warehouse/warehouse/${warehouseId}/location-recommendation`,
    method: 'get',
    params
  })
}

export function getDemandForecastAPI(warehouseId: number, params: { skuCode?: string; days?: number }) {
  return http<{ forecasts: Array<{ date: string; quantity: number; confidence: number }> }>({
    url: `/api/ai-warehouse/warehouse/${warehouseId}/demand-forecast`,
    method: 'get',
    params
  })
}

export function getRestockSuggestionAPI(warehouseId: number, params: { skuCode?: string }) {
  return http<{ suggestions: Array<{ skuCode: string; quantity: number; urgency: string }> }>({
    url: `/api/ai-warehouse/warehouse/${warehouseId}/restock-suggestion`,
    method: 'get',
    params
  })
}

export function detectAnomaliesAPI(warehouseId: number, params: { startDate?: string; endDate?: string }) {
  return http<{ anomalies: Array<{ id: number; type: string; description: string; severity: string; detectedAt: string }> }>({
    url: `/api/ai-warehouse/warehouse/${warehouseId}/anomaly-detection`,
    method: 'get',
    params
  })
}