import type { CommonPage } from '@/types/common'
import type {
  CouponHistoryQueryParam,
  CouponQueryParam,
  SmsCoupon,
  SmsCouponExt,
} from '@/types/coupon'
import http from '@/utils/http'
export function getCouponListAPI(params: CouponQueryParam) {
  return http<CommonPage<SmsCoupon>>({
    url: '/coupon/list',
    method: 'get',
    params: params,
  })
}
export function couponCreateAPI(data: SmsCouponExt) {
  return http({
    url: '/coupon/create',
    method: 'post',
    data: data,
  })
}
export function getCouponByIdAPI(id: number) {
  return http<SmsCouponExt>({
    url: '/coupon/' + id,
    method: 'get',
  })
}
export function couponUpdateByIdAPI(id: number, data: SmsCouponExt) {
  return http({
    url: '/coupon/update/' + id,
    method: 'post',
    data: data,
  })
}
export function couponDeleteByIdAPI(id: number) {
  return http({
    url: '/coupon/delete/' + id,
    method: 'post',
  })
}
export function getCouponHistoryListAPI(params: CouponHistoryQueryParam) {
  return http({
    url: '/couponHistory/list',
    method: 'get',
    params: params,
  })
}
