import type { PageParam } from './common'
import type { PmsProduct } from './product'

/** Flash sale activity info */
export type SmsFlashPromotion = {
  /** ID */
  id?: number
  /** Flash sale time slot name */
  title: string
  /** Start date */
  startDate?: string
  /** End date */
  endDate?: string
  /** Online/offline status */
  status: number
  /** Creation time */
  createTime?: string
}

/** Flash sale time slot info */
export type SmsFlashPromotionSession = {
  /** Number */
  id?: number
  /** Session name */
  name: string
  /** Daily start time */
  startTime?: string
  /** Daily end time */
  endTime?: string
  /** Enabled status: 0 -> disabled; 1 -> enabled */
  status: number
  /** Creation time */
  createTime?: string
  /** Product quantity */
  productCount?: number
}

/** Flash sale activity-product relation */
export type SmsFlashPromotionProductRelation = {
  /** Number */
  id?: number
  /** Flash sale activity ID */
  flashPromotionId: number
  /** Flash sale time slot ID */
  flashPromotionSessionId: number
  /** Product ID */
  productId: number
  /** Flash sale price */
  flashPromotionPrice?: number
  /** Flash sale quantity */
  flashPromotionCount?: number
  /** Max quantity per person */
  flashPromotionLimit?: number
  /** Sort order */
  sort?: number
  /** Associated product */
  product?: PmsProduct
}

/** Flash sale product list query params */
export type FlashProductQueryParam = PageParam & {
  /** Flash sale activity ID */
  flashPromotionId?: number
  /** Flash sale session ID */
  flashPromotionSessionId?: number
}
