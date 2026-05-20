import type { SmsFlashPromotionSession } from '@/types/flash'
import http from '@/utils/http'
export function getFlashSessionSelectListAPI(params: { flashPromotionId: number }) {
  return http<SmsFlashPromotionSession[]>({
    url: '/flashSession/selectList',
    method: 'get',
    params: params,
  })
}
export function getFlashSessionListAPI() {
  return http<SmsFlashPromotionSession[]>({
    url: '/flashSession/list',
    method: 'get',
  })
}
export function flashSessionUpdateStatusByIdAPI(id: number, params: { status: number }) {
  return http({
    url: '/flashSession/update/status/' + id,
    method: 'post',
    params: params,
  })
}
export function flashSessionDeleteByIdAPI(id: number) {
  return http({
    url: '/flashSession/delete/' + id,
    method: 'post',
  })
}
export function flashSessionCreateAPI(data: SmsFlashPromotionSession) {
  return http({
    url: '/flashSession/create',
    method: 'post',
    data: data,
  })
}
export function flashSessionUpdateByIdAPI(id: number, data: SmsFlashPromotionSession) {
  return http({
    url: '/flashSession/update/' + id,
    method: 'post',
    data: data,
  })
}
