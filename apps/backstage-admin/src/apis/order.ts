import type { CommonPage } from '@/types/common'
import type {
  OmsMoneyInfoParam,
  OmsOrder,
  OmsOrderDeliveryParam,
  OmsOrderDetail,
  OmsReceiverInfoParam,
  OrderQueryParam,
} from '@/types/order'
import http from '@/utils/http'
export function getOrderListAPI(params: OrderQueryParam) {
  return http<CommonPage<OmsOrder>>({
    url: '/order/list',
    method: 'get',
    params: params,
  })
}
export function orderUpdateCloseAPI(params: { ids: string; note: string }) {
  return http({
    url: '/order/update/close',
    method: 'post',
    params: params,
  })
}
export function orderDeleteByIdsAPI(params: { ids: string }) {
  return http({
    url: '/order/delete',
    method: 'post',
    params: params,
  })
}
export function orderUpdateDeliveryAPI(data: OmsOrderDeliveryParam[]) {
  return http({
    url: '/order/update/delivery',
    method: 'post',
    data: data,
  })
}
export function getOrderDetailByIdAPI(id: number) {
  return http<OmsOrderDetail>({
    url: '/order/' + id,
    method: 'get',
  })
}
export function orderUpdateReceiverInfoAPI(data: OmsReceiverInfoParam) {
  return http({
    url: '/order/update/receiverInfo',
    method: 'post',
    data: data,
  })
}
export function orderUpdateMoneyInfoAPI(data: OmsMoneyInfoParam) {
  return http({
    url: '/order/update/moneyInfo',
    method: 'post',
    data: data,
  })
}
export function orderUpdateNoteAPI(params: { id: number; note: string; status: number }) {
  return http({
    url: '/order/update/note',
    method: 'post',
    params: params,
  })
}
