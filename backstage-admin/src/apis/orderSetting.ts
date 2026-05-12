import type { OmsOrderSetting } from '@/types/orderSetting'
import http from '@/utils/http'
export function getOrderSettingByIdAPI(id: number) {
  return http<OmsOrderSetting>({
    url: '/orderSetting/' + id,
    method: 'get',
  })
}
export function orderSettingUpdateByIdAPI(id: number, data: OmsOrderSetting) {
  return http({
    url: '/orderSetting/update/' + id,
    method: 'post',
    data: data,
  })
}
