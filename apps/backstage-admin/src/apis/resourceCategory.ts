import type { UmsResourceCategory } from '@/types/resource'
import http from '@/utils/http'
export function resourceCategoryListAllAPI() {
  return http<UmsResourceCategory[]>({
    url: '/resourceCategory/listAll',
    method: 'get',
  })
}
export function resourceCategoryCreateAPI(data: UmsResourceCategory) {
  return http({
    url: '/resourceCategory/create',
    method: 'post',
    data: data,
  })
}
export function resourceCategoryUpdateAPI(id: number, data: UmsResourceCategory) {
  return http({
    url: '/resourceCategory/update/' + id,
    method: 'post',
    data: data,
  })
}
export function resourceCategoryDeleteByIdAPI(id: number) {
  return http({
    url: '/resourceCategory/delete/' + id,
    method: 'post',
  })
}
