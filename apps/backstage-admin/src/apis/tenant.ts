import http from '@/utils/http'

export interface Tenant {
  id?: number
  name?: string
  code?: string
  status?: string
  contactName?: string
  contactEmail?: string
  contactPhone?: string
  domain?: string
  config?: string
  createdAt?: string
  updatedAt?: string
}

export interface TenantShop {
  id?: number
  tenantId?: number
  name?: string
  description?: string
  logo?: string
  banner?: string
  status?: string
  sortOrder?: number
  createdAt?: string
  updatedAt?: string
}

export interface TenantQueryParam {
  pageNum?: number
  pageSize?: number
  status?: string
}

export function getTenantListAPI(params: TenantQueryParam) {
  return http<Tenant[]>({
    url: '/tenant',
    method: 'get',
    params: { page: params.pageNum || 1, size: params.pageSize || 10, status: params.status },
  })
}

export function getTenantByIdAPI(id: number) {
  return http<Tenant>({
    url: '/tenant/' + id,
    method: 'get',
  })
}

export function approveTenantAPI(id: number) {
  return http<void>({
    url: '/tenant/' + id + '/approve',
    method: 'post',
  })
}

export function rejectTenantAPI(id: number) {
  return http<void>({
    url: '/tenant/' + id + '/reject',
    method: 'post',
  })
}

export function disableTenantAPI(id: number) {
  return http<void>({
    url: '/tenant/' + id + '/disable',
    method: 'post',
  })
}

export function getShopByTenantAPI(tenantId: number) {
  return http<TenantShop>({
    url: '/tenant/' + tenantId + '/shop',
    method: 'get',
  })
}

export function updateShopAPI(tenantId: number, data: TenantShop) {
  return http<TenantShop>({
    url: '/tenant/' + tenantId + '/shop',
    method: 'put',
    data: data,
  })
}
