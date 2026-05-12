import type { CommonPage } from '@/types/common'
import type { HomeAdvertiseQueryParam, SmsHomeAdvertise } from '@/types/homeAdvertist'
import http from '@/utils/http'
export function getHomeAdvertiseListAPI(params: HomeAdvertiseQueryParam) {
  return http<CommonPage<SmsHomeAdvertise>>({
    url: '/home/advertise/list',
    method: 'get',
    params: params,
  })
}
export function homeAdvertiseUpdateStatusAPI(params: { id: number; status: number }) {
  return http({
    url: '/home/advertise/update/status/' + params.id,
    method: 'post',
    params: params,
  })
}
export function deleteHomeAdvertiseAPI(params: { ids: string }) {
  return http({
    url: '/home/advertise/delete',
    method: 'post',
    params: params,
  })
}
export function homeAdvertiseCreateAPI(data: SmsHomeAdvertise) {
  return http({
    url: '/home/advertise/create',
    method: 'post',
    data: data,
  })
}
export function getHomeAdvertiseByIdAPI(id: number) {
  return http({
    url: '/home/advertise/' + id,
    method: 'get',
  })
}
export function homeAdvertiseUpdateAPI(id: number, data: SmsHomeAdvertise) {
  return http({
    url: '/home/advertise/update/' + id,
    method: 'post',
    data: data,
  })
}
