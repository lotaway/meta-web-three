import type { CommonPage } from '@/types/common'
import type { FlashProductQueryParam, SmsFlashPromotionProductRelation } from '@/types/flash'
import http from '@/utils/http'
export function getFlashProductRelationListAPI(params: FlashProductQueryParam) {
  return http<CommonPage<SmsFlashPromotionProductRelation>>({
    url: '/flashProductRelation/list',
    method: 'get',
    params: params,
  })
}
export function flashProductRelationCreateAPI(data: SmsFlashPromotionProductRelation[]) {
  return http({
    url: '/flashProductRelation/create',
    method: 'post',
    data: data,
  })
}
export function flashProductRelationDeleteByIdAPI(id: number) {
  return http({
    url: '/flashProductRelation/delete/' + id,
    method: 'post',
  })
}
export function flashProductRelationUpdateByIdAPI(
  id: number,
  data: SmsFlashPromotionProductRelation,
) {
  return http({
    url: '/flashProductRelation/update/' + id,
    method: 'post',
    data: data,
  })
}
