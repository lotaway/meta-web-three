import type { PmsBrand } from '@/types/brand'
import type { CommonPage, PageParam } from '@/types/common'
import http from '@/utils/http'
export function getBrandListAPI(params: PageParam) {
  return http<CommonPage<PmsBrand>>({
    url: '/brand/list',
    method: 'get',
    params: params,
  })
}
export function createBrandAPI(data: PmsBrand) {
  return http({
    url: '/brand/create',
    method: 'post',
    data: data,
  })
}
export function brandUpdateShowStatusAPI(params: { ids: string; showStatus: number }) {
  return http({
    url: '/brand/update/showStatus',
    method: 'post',
    params: params,
  })
}
export function brandUpdateFactoryStatusAPI(params: { ids: string; factoryStatus: number }) {
  return http({
    url: '/brand/update/factoryStatus',
    method: 'post',
    params: params,
  })
}
export function brandDeleteByIdAPI(id: number) {
  return http({
    url: '/brand/delete/' + id,
    method: 'get',
  })
}
export function getBrandAPI(id: number) {
  return http<PmsBrand>({
    url: '/brand/' + id,
    method: 'get',
  })
}
export function updateBrandAPI(id: number, data: PmsBrand) {
  return http({
    url: '/brand/update/' + id,
    method: 'post',
    data: data,
  })
}
