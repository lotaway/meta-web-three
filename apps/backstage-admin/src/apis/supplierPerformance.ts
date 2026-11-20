import http from '@/utils/http'

export interface SupplierPerformance {
  id?: number
  supplierId?: number
  supplierCode?: string
  supplierName?: string
  periodStart?: string
  periodEnd?: string
  onTimeDeliveryRate?: number
  qualityPassRate?: number
  priceCompetitivenessScore?: number
  overallScore?: number
  assessmentLevel?: string
  totalOrders?: number
  onTimeDeliveryCount?: number
  qualifiedCount?: number
  totalQualityCheckCount?: number
  marketAvgPrice?: number
  supplierPrice?: number
  remark?: string
  assessor?: string
  assessmentDate?: string
  createdAt?: string
  updatedAt?: string
}

export interface SupplierPerformanceDashboard {
  totalSuppliers: number
  levelACount: number
  levelBCount: number
  levelCCount: number
  levelDCount: number
  avgOnTimeDeliveryRate: number
  avgQualityPassRate: number
  avgPriceCompetitivenessScore: number
  avgOverallScore: number
  improvementNeededSuppliers: SupplierPerformance[]
}

// Get all performance evaluation records
export function getAllSupplierPerformanceAPI() {
  return http<SupplierPerformance[]>({ url: '/api/supplier-performance', method: 'get' })
}

// Query performance evaluation records by supplier ID
export function getSupplierPerformanceBySupplierIdAPI(supplierId: number) {
  return http<SupplierPerformance[]>({ url: `/api/supplier-performance/supplier/${supplierId}`, method: 'get' })
}

// Query performance evaluation records by ID
export function getSupplierPerformanceByIdAPI(id: number) {
  return http<SupplierPerformance>({ url: `/api/supplier-performance/${id}`, method: 'get' })
}

// Query performance evaluation records by rating level
export function getSupplierPerformanceByLevelAPI(level: string) {
  return http<SupplierPerformance[]>({ url: `/api/supplier-performance/level/${level}`, method: 'get' })
}

// Create or update a performance evaluation record
export function createOrUpdateSupplierPerformanceAPI(data: SupplierPerformance) {
  return http<SupplierPerformance>({ url: '/api/supplier-performance', method: 'post', data })
}

// Delete a performance evaluation record
export function deleteSupplierPerformanceAPI(id: number) {
  return http<null>({ url: `/api/supplier-performance/${id}`, method: 'delete' })
}

// Get performance evaluation dashboard data
export function getSupplierPerformanceDashboardAPI() {
  return http<SupplierPerformanceDashboard>({ url: '/api/supplier-performance/dashboard', method: 'get' })
}