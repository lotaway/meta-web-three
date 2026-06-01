import type { CommonPage } from '@/types/common'
import type { SmsCoupon, SmsCouponHistory, CouponQueryParam, CouponHistoryQueryParam } from '@/types/coupon'
import type { SmsFlashPromotion, SmsFlashPromotionSession, SmsFlashPromotionProductRelation, FlashProductQueryParam } from '@/types/flash'
import type { PmsHomeBrand } from '@/types/homeBrand'
import type { PmsHomeSubject } from '@/types/homeSubject'
import type { PmsHomeNewProduct } from '@/types/newProduct'
import http from '@/utils/http'

// Coupon APIs
export function getCouponListAPI(params: CouponQueryParam) {
  return http<CommonPage<SmsCoupon>>({
    url: '/coupon/list',
    method: 'get',
    params: params,
  })
}

export function couponCreateAPI(data: SmsCoupon) {
  return http({
    url: '/coupon/create',
    method: 'post',
    data: data,
  })
}

export function getCouponByIdAPI(id: number) {
  return http<SmsCoupon>({
    url: '/coupon/' + id,
    method: 'get',
  })
}

export function couponUpdateByIdAPI(id: number, data: SmsCoupon) {
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
  return http<CommonPage<SmsCouponHistory>>({
    url: '/couponHistory/list',
    method: 'get',
    params: params,
  })
}

// Flash Promotion APIs
export function getFlashPromotionListAPI(params: { pageNum: number; pageSize: number; keyword?: string }) {
  return http<CommonPage<SmsFlashPromotion>>({
    url: '/flash/list',
    method: 'get',
    params: params,
  })
}

export function flashPromotionCreateAPI(data: SmsFlashPromotion) {
  return http({
    url: '/flash/create',
    method: 'post',
    data: data,
  })
}

export function flashPromotionUpdateAPI(id: number, data: SmsFlashPromotion) {
  return http({
    url: '/flash/update/' + id,
    method: 'post',
    data: data,
  })
}

export function flashPromotionUpdateStatusAPI(id: number, status: number) {
  return http({
    url: '/flash/update/status/' + id,
    method: 'post',
    params: { status },
  })
}

export function flashPromotionDeleteAPI(id: number) {
  return http({
    url: '/flash/delete/' + id,
    method: 'post',
  })
}

// Flash Session APIs
export function getFlashSessionListAPI(params: { pageNum: number; pageSize: number }) {
  return http<CommonPage<SmsFlashPromotionSession>>({
    url: '/flashSession/list',
    method: 'get',
    params: params,
  })
}

export function flashSessionCreateAPI(data: SmsFlashPromotionSession) {
  return http({
    url: '/flashSession/create',
    method: 'post',
    data: data,
  })
}

export function flashSessionUpdateAPI(id: number, data: SmsFlashPromotionSession) {
  return http({
    url: '/flashSession/update/' + id,
    method: 'post',
    data: data,
  })
}

export function flashSessionDeleteAPI(id: number) {
  return http({
    url: '/flashSession/delete/' + id,
    method: 'post',
  })
}

// Flash Product Relation APIs
export function getFlashProductRelationListAPI(params: FlashProductQueryParam) {
  return http<CommonPage<SmsFlashPromotionProductRelation>>({
    url: '/flashProductRelation/list',
    method: 'get',
    params: params,
  })
}

export function flashProductRelationCreateAPI(data: SmsFlashPromotionProductRelation) {
  return http({
    url: '/flashProductRelation/create',
    method: 'post',
    data: data,
  })
}

export function flashProductRelationUpdateAPI(id: number, data: SmsFlashPromotionProductRelation) {
  return http({
    url: '/flashProductRelation/update/' + id,
    method: 'post',
    data: data,
  })
}

export function flashProductRelationDeleteAPI(id: number) {
  return http({
    url: '/flashProductRelation/delete/' + id,
    method: 'post',
  })
}

// Home Brand APIs
export function getHomeBrandListAPI(params: { pageNum: number; pageSize: number; brandName?: string; recommendStatus?: number }) {
  return http<CommonPage<PmsHomeBrand>>({
    url: '/home/brand/list',
    method: 'get',
    params: params,
  })
}

export function homeBrandCreateAPI(data: PmsHomeBrand) {
  return http({
    url: '/home/brand/create',
    method: 'post',
    data: data,
  })
}

export function homeBrandUpdateAPI(id: number, data: PmsHomeBrand) {
  return http({
    url: '/home/brand/update/' + id,
    method: 'post',
    data: data,
  })
}

export function homeBrandUpdateRecommendStatusAPI(id: number, recommendStatus: number) {
  return http({
    url: '/home/brand/update/recommendStatus',
    method: 'post',
    params: { id, recommendStatus },
  })
}

export function homeBrandDeleteAPI(id: number) {
  return http({
    url: '/home/brand/delete/' + id,
    method: 'post',
  })
}

// Home Subject APIs
export function getHomeSubjectListAPI(params: { pageNum: number; pageSize: number; subjectName?: string; recommendStatus?: number }) {
  return http<CommonPage<PmsHomeSubject>>({
    url: '/home/subject/list',
    method: 'get',
    params: params,
  })
}

export function homeSubjectCreateAPI(data: PmsHomeSubject) {
  return http({
    url: '/home/subject/create',
    method: 'post',
    data: data,
  })
}

export function homeSubjectUpdateAPI(id: number, data: PmsHomeSubject) {
  return http({
    url: '/home/subject/update/' + id,
    method: 'post',
    data: data,
  })
}

export function homeSubjectUpdateRecommendStatusAPI(id: number, recommendStatus: number) {
  return http({
    url: '/home/subject/update/recommendStatus',
    method: 'post',
    params: { id, recommendStatus },
  })
}

export function homeSubjectDeleteAPI(id: number) {
  return http({
    url: '/home/subject/delete/' + id,
    method: 'post',
  })
}

// Home New Product APIs
export function getHomeNewProductListAPI(params: { pageNum: number; pageSize: number; productName?: string; recommendStatus?: number }) {
  return http<CommonPage<PmsHomeNewProduct>>({
    url: '/home/newProduct/list',
    method: 'get',
    params: params,
  })
}

export function homeNewProductCreateAPI(data: PmsHomeNewProduct) {
  return http({
    url: '/home/newProduct/create',
    method: 'post',
    data: data,
  })
}

export function homeNewProductUpdateAPI(id: number, data: PmsHomeNewProduct) {
  return http({
    url: '/home/newProduct/update/' + id,
    method: 'post',
    data: data,
  })
}

export function homeNewProductUpdateRecommendStatusAPI(id: number, recommendStatus: number) {
  return http({
    url: '/home/newProduct/update/recommendStatus',
    method: 'post',
    params: { id, recommendStatus },
  })
}

export function homeNewProductDeleteAPI(id: number) {
  return http({
    url: '/home/newProduct/delete/' + id,
    method: 'post',
  })
}