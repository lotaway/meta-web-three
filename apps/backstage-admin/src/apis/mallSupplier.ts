import http from '@/utils/http'

export interface MallSupplier {
  id?: number
  supplierCode: string
  supplierName: string
  contactPerson: string
  contactPhone: string
  contactEmail: string
  address: string
  status: number
  verificationStatus: number
  businessLicense: string
  legalPerson: string
  supplierLevel: number
  score: number
  remark: string
  createTime?: string
  updateTime?: string
}

export interface MallSupplierRegistration {
  supplierName: string
  contactPerson: string
  contactPhone: string
  contactEmail: string
  address: string
  businessLicense: string
  legalPerson: string
}

export interface MallSupplierVerification {
  supplierId: number
  approved: boolean
  reason: string
}

export interface MallSupplierPerformance {
  supplierId: number
  orderCount: number
  onTimeDeliveryRate: number
  qualityScore: number
  responseTime: number
  overallScore: number
}

export function getMallSupplierListAPI() {
  return http<MallSupplier[]>({
    url: '/api/supplier/list',
    method: 'get',
  })
}

export function getMallSupplierByIdAPI(id: number) {
  return http<MallSupplier>({
    url: `/api/supplier/${id}`,
    method: 'get',
  })
}

export function registerMallSupplierAPI(data: MallSupplierRegistration) {
  return http<MallSupplier>({
    url: '/api/supplier/register',
    method: 'post',
    data,
  })
}

export function submitMallSupplierVerificationAPI(id: number) {
  return http<MallSupplier>({
    url: `/api/supplier/${id}/submit-verification`,
    method: 'post',
  })
}

export function verifyMallSupplierAPI(data: MallSupplierVerification) {
  return http<MallSupplier>({
    url: '/api/supplier/verify',
    method: 'post',
    data,
  })
}

export function getMallSupplierPerformanceAPI(id: number) {
  return http<MallSupplierPerformance>({
    url: `/api/supplier/${id}/performance`,
    method: 'get',
  })
}

export function updateMallSupplierScoreAPI(id: number, delta: number) {
  return http<MallSupplier>({
    url: `/api/supplier/${id}/score`,
    method: 'post',
    params: { delta },
  })
}
