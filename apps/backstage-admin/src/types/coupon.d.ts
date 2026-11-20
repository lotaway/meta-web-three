import type { PageParam } from './common'

/** Coupon info */
export type SmsCoupon = {
  /** ID */
  id?: number
  /** Coupon type: 0 -> all-users gift; 1 -> member gift; 2 -> purchase gift; 3 -> registration gift */
  type: number
  /** Coupon name */
  name: string
  /** Usage platform: 0 -> all; 1 -> mobile; 2 -> PC */
  platform: number
  /** Quantity */
  count?: number
  /** Amount */
  amount: number
  /** Max coupons per person */
  perLimit: number
  /** Usage threshold; 0 means no threshold */
  minPoint?: number
  /** Start time */
  startTime?: string
  /** End time */
  endTime?: string
  /** Usage type: 0 -> all products; 1 -> specific category; 2 -> specific product */
  useType: number
  /** Remark */
  note?: string
  /** Issued quantity */
  publishCount?: number
  /** Used quantity */
  useCount?: number
  /** Claimed quantity */
  receiveCount?: number
  /** Available claim date */
  enableTime?: string
  /** Promo code */
  code?: string
  /** Eligible member type: 0 -> unlimited */
  memberLevel?: number
}

/** Coupon extension */
export type SmsCouponExt = SmsCoupon & {
  /** Coupon-bound product */
  productRelationList?: CouponSelectProductOptionVo[]
  /** Coupon-bound product category */
  productCategoryRelationList?: CouponProductCateRelationVo[]
}

/** Coupon-product category relation */
export type CouponProductCateRelationVo = {
  /** Product category ID */
  productCategoryId?: number
  /** Product category name */
  productCategoryName?: string
  /** Parent category name */
  parentCategoryName?: string
}

/** Coupon designated product selection object */
export type CouponSelectProductOptionVo = {
  /** Product ID */
  productId?: number
  /** Product name */
  productName: string
  /** Product SN code */
  productSn: string
}

/** Coupon list query params */
export type CouponQueryParam = PageParam & {
  /** Query name */
  name?: string
  /** Coupon type */
  type?: number
}

/** Coupon history record */
export type SmsCouponHistory = {
  /** ID */
  id?: number
  /** Coupon ID */
  couponId?: number
  /** Member ID */
  memberId?: number
  /** Promo code */
  couponCode?: string
  /** Claimer nickname */
  memberNickname?: string
  /** Acquisition type: 0 -> admin grant; 1 -> self claim */
  getType?: number
  /** Creation time */
  createTime?: string
  /** Usage status: 0 -> unused; 1 -> used; 2 -> expired */
  useStatus?: number
  /** Usage time */
  useTime?: string
  /** Order ID */
  orderId?: number
  /** Order number */
  orderSn?: string
}

/** Coupon history list query params */
export type CouponHistoryQueryParam = PageParam & {
  /** Usage status: 0 -> unused; 1 -> used; 2 -> expired */
  useStatus?: number
  /** Order number */
  orderSn?: string
  /** Coupon ID */
  couponId?: number
}
