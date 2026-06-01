import http from '@/utils/http'

export interface Supplier {
  id?: number
  supplierCode: string
  supplierName: string
  name?: string
  supplierType?: string
  province?: string
  city?: string
  address?: string
  contact?: string
  contactPerson?: string
  contactPhone?: string
  phone?: string
  email?: string
  status?: string
  creditLimit?: number
  paymentTerms?: string
  category?: string
  score?: number
  level?: string
  assessmentLevel?: string
  createdAt?: string
  updatedAt?: string
}

export interface SupplierQueryParam {
  pageNum?: number
  pageSize?: number
  status?: string
  category?: string
  supplierCode?: string
  supplierName?: string
}

export function getSupplierListAPI(params: SupplierQueryParam) {
  return http<Supplier[]>({
    url: '/api/supplier/suppliers',
    method: 'get',
    params: params,
  })
}

export function getSupplierByIdAPI(id: number) {
  return http<Supplier>({
    url: '/api/supplier/suppliers/' + id,
    method: 'get',
  })
}

export function getSupplierByCodeAPI(code: string) {
  return http<Supplier>({
    url: '/api/supplier/suppliers/code/' + code,
    method: 'get',
  })
}

export function createSupplierAPI(data: Supplier) {
  return http<Supplier>({
    url: '/api/supplier/suppliers',
    method: 'post',
    data: data,
  })
}

export function updateSupplierAPI(id: number, data: Supplier) {
  return http<Supplier>({
    url: '/api/supplier/suppliers/' + id,
    method: 'put',
    data: data,
  })
}

export function updateSupplierAssessmentAPI(id: number, level: string) {
  return http<Supplier>({
    url: '/api/supplier/suppliers/' + id + '/assessment',
    method: 'put',
    params: { level },
  })
}