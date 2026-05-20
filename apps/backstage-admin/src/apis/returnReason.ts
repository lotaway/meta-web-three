import type { CommonPage, PageParam } from '@/types/common'
import type { OmsOrderReturnReason } from '@/types/returnReason'
import http from '@/utils/http'
export function getReturnReasonListAPI(params: PageParam) {
  return http<CommonPage<OmsOrderReturnReason>>({
    url: '/returnReason/list',
    method: 'get',
    params: params,
  })
}
export function returnReasonDeleteByIdsAPI(params: { ids: string }) {
  return http({
    url: '/returnReason/delete',
    method: 'post',
    params: params,
  })
}
export function returnReasonUpdateStatusAPI(params: { ids: string; status: number }) {
  return http({
    url: '/returnReason/update/status',
    method: 'post',
    params: params,
  })
}
export function returnReasonCreateAPI(data: OmsOrderReturnReason) {
  return http({
    url: '/returnReason/create',
    method: 'post',
    data: data,
  })
}
export function getReturnReasonByIdAPI(id: number) {
  return http<OmsOrderReturnReason>({
    url: '/returnReason/' + id,
    method: 'get',
  })
}
export function returnReasonUpdateAPI(id: number, data: OmsOrderReturnReason) {
  return http({
    url: '/returnReason/update/' + id,
    method: 'post',
    data: data,
  })
}
