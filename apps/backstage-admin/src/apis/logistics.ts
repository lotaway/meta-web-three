import http from '@/utils/http'

export interface LogisticsOrder {
  id?: number
  trackingNo: string
  orderNo: string
  carrierId?: number
  carrierName?: string
  status: string
  senderName: string
  senderPhone: string
  senderAddress: string
  receiverName: string
  receiverPhone: string
  receiverAddress: string
  weight?: number
  volume?: number
  freightFee?: number
  createTime?: string
  updateTime?: string
}

export interface LogisticsQueryParam {
  pageNum?: number
  pageSize?: number
  carrierId?: number
  status?: string
  trackingNo?: string
  orderNo?: string
}

export function getLogisticsListAPI(params: LogisticsQueryParam) {
  return http<LogisticsOrder[]>({
    url: '/api/logistics/orders',
    method: 'get',
    params: params,
  })
}

export function getLogisticsByTrackingNoAPI(trackingNo: string) {
  return http<LogisticsOrder>({
    url: '/api/logistics/track/' + trackingNo,
    method: 'get',
  })
}

export function getLogisticsByOrderNoAPI(orderNo: string) {
  return http<LogisticsOrder>({
    url: '/api/logistics/order/' + orderNo,
    method: 'get',
  })
}

export function createLogisticsOrderAPI(data: LogisticsOrder) {
  return http<LogisticsOrder>({
    url: '/api/logistics/orders',
    method: 'post',
    data: data,
  })
}

export function updateLogisticsStatusAPI(trackingNo: string, status: string) {
  return http<LogisticsOrder>({
    url: '/api/logistics/orders/' + trackingNo + '/status',
    method: 'put',
    params: { status },
  })
}