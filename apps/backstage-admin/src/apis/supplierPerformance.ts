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

// 获取所有绩效评估记录
export function getAllSupplierPerformanceAPI() {
  return http<SupplierPerformance[]>({ url: '/api/supplier-performance', method: 'get' })
}

// 根据供应商ID查询绩效评估记录
export function getSupplierPerformanceBySupplierIdAPI(supplierId: number) {
  return http<SupplierPerformance[]>({ url: `/api/supplier-performance/supplier/${supplierId}`, method: 'get' })
}

// 根据ID查询绩效评估记录
export function getSupplierPerformanceByIdAPI(id: number) {
  return http<SupplierPerformance>({ url: `/api/supplier-performance/${id}`, method: 'get' })
}

// 根据评估等级查询绩效评估记录
export function getSupplierPerformanceByLevelAPI(level: string) {
  return http<SupplierPerformance[]>({ url: `/api/supplier-performance/level/${level}`, method: 'get' })
}

// 创建或更新绩效评估记录
export function createOrUpdateSupplierPerformanceAPI(data: SupplierPerformance) {
  return http<SupplierPerformance>({ url: '/api/supplier-performance', method: 'post', data })
}

// 删除绩效评估记录
export function deleteSupplierPerformanceAPI(id: number) {
  return http<null>({ url: `/api/supplier-performance/${id}`, method: 'delete' })
}

// 获取绩效评估看板数据
export function getSupplierPerformanceDashboardAPI() {
  return http<SupplierPerformanceDashboard>({ url: '/api/supplier-performance/dashboard', method: 'get' })
}