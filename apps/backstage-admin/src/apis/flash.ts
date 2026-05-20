import type { PageParam } from '@/types/common'
import type { SmsFlashPromotion } from '@/types/flash'
import http from '@/utils/http'
export function getFlashListAPI(params: PageParam) {
  return http({
    url: '/flash/list',
    method: 'get',
    params: params,
  })
}
export function flashUpdateStatusByIdAPI(id: number, params: { status: number }) {
  return http({
    url: '/flash/update/status/' + id,
    method: 'post',
    params: params,
  })
}
export function flashDeleteByIdAPI(id: number) {
  return http({
    url: '/flash/delete/' + id,
    method: 'post',
  })
}
export function flashCreateAPI(data: SmsFlashPromotion) {
  return http({
    url: '/flash/create',
    method: 'post',
    data: data,
  })
}
export function flashUpdateByIdAPI(id: number, data: SmsFlashPromotion) {
  return http({
    url: '/flash/update/' + id,
    method: 'post',
    data: data,
  })
}
