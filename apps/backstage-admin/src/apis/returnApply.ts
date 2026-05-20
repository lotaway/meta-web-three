import type { CommonPage } from '@/types/common'
import type {
  OmsOrderReturnApply,
  OmsOrderReturnApplyResult,
  OmsUpdateStatusParam,
  ReturnApplyQueryParam,
} from '@/types/returnApply'
import http from '@/utils/http'
export function getReturnApplyListAPI(params: ReturnApplyQueryParam) {
  return http<CommonPage<OmsOrderReturnApply>>({
    url: '/returnApply/list',
    method: 'get',
    params: params,
  })
}
export function returnApplyDeleteByIdsAPI(params: { ids: string }) {
  return http({
    url: '/returnApply/delete',
    method: 'post',
    params: params,
  })
}
export function returnApplyUpdateStatusAPI(id: number, data: OmsUpdateStatusParam) {
  return http({
    url: '/returnApply/update/status/' + id,
    method: 'post',
    data: data,
  })
}
export function getReturnApplyByIdAPI(id: number) {
  return http<OmsOrderReturnApplyResult>({
    url: '/returnApply/' + id,
    method: 'get',
  })
}
