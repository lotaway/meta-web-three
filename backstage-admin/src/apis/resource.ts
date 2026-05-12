import type { CommonPage } from '@/types/common'
import type { ResourceQueryParam, UmsResource } from '@/types/resource'
import http from '@/utils/http'
export function fetchAllResourceList() {
  return http<UmsResource[]>({
    url: '/resource/listAll',
    method: 'get',
  })
}
export function getResourceListAPI(params: ResourceQueryParam) {
  return http<CommonPage<UmsResource>>({
    url: '/resource/list',
    method: 'get',
    params: params,
  })
}
export function resourceCreateAPI(data: UmsResource) {
  return http({
    url: '/resource/create',
    method: 'post',
    data: data,
  })
}
export function resourceUpdateAPI(id: number, data: UmsResource) {
  return http({
    url: '/resource/update/' + id,
    method: 'post',
    data: data,
  })
}
export function resourceDeleteByIdAPI(id: number) {
  return http({
    url: '/resource/delete/' + id,
    method: 'post',
  })
}
