import type { PageParam } from './common'

/** Order info */
export type OmsOrder = {
  /** Order ID */
  id: number
  /** Member ID */
  memberId?: number
  /** Coupon ID */
  couponId?: number
  /** Order number */
  orderSn?: string
  /** Submission time */
  createTime: string
  /** User account */
  memberUsername?: string
  /** Total order amount */
  totalAmount: number
  /** Payable amount (actual paid amount) */
  payAmount: number
  /** Shipping fee amount */
  freightAmount: number
  /** Promotion discount amount (promo price, full-reduction, tiered price) */
  promotionAmount?: number
  /** Points deduction amount */
  integrationAmount?: number
  /** Coupon deduction amount */
  couponAmount?: number
  /** Discount amount adjusted by admin in backend */
  discountAmount: number
  /** Payment method: 0 -> unpaid; 1 -> Alipay; 2 -> WeChat */
  payType: number
  /** Order source: 0 -> PC order; 1 -> app order */
  sourceType: number
  /** Order status: 0 -> pending payment; 1 -> pending shipment; 2 -> shipped; 3 -> completed; 4 -> closed; 5 -> invalid */
  status: number
  /** Order type: 0 -> normal order; 1 -> flash sale order */
  orderType: number
  /** Shipping company (delivery method) */
  deliveryCompany?: string
  /** Tracking number */
  deliverySn?: string
  /** Auto-confirm time (days) */
  autoConfirmDay?: number
  /** Points earned */
  integration?: number
  /** Growth value earned */
  growth?: number
  /** Activity info */
  promotionInfo: string
  /** Invoice type: 0 -> none; 1 -> electronic; 2 -> paper */
  billType?: number
  /** Invoice title */
  billHeader?: string
  /** Invoice content */
  billContent?: string
  /** Invoice recipient phone */
  billReceiverPhone?: string
  /** Invoice recipient email */
  billReceiverEmail?: string
  /** Recipient name */
  receiverName?: string
  /** Recipient phone */
  receiverPhone?: string
  /** Recipient postal code */
  receiverPostCode?: string
  /** Province / municipality */
  receiverProvince: string
  /** City */
  receiverCity?: string
  /** District */
  receiverRegion?: string
  /** Detailed address */
  receiverDetailAddress?: string
  /** Order remark */
  note?: string
  /** Receipt confirmation status: 0 -> unconfirmed; 1 -> confirmed */
  confirmStatus?: number
  /** Deletion status: 0 -> not deleted; 1 -> deleted */
  deleteStatus?: number
  /** Points used at order placement */
  useIntegration?: number
  /** Payment time */
  paymentTime: string
  /** Shipment time */
  deliveryTime: string
  /** Receipt confirmation time */
  receiveTime: string
  /** Review time */
  commentTime: string
  /** Modification time */
  modifyTime?: string
}

/** Order item */
export type OmsOrderItem = {
  /** Order item ID */
  id: number
  /** Order ID */
  orderId: number
  /** Order number */
  orderSn: string
  /** Product ID */
  productId: number
  /** Product image */
  productPic: string
  /** Product name */
  productName: string
  /** Product brand */
  productBrand: string
  /** Product SKU number */
  productSn: string
  /** Sale price */
  productPrice: number
  /** Purchase quantity */
  productQuantity: number
  /** Product SKU number */
  productSkuId: number
  /** Product SKU barcode */
  productSkuCode: string
  /** Product category ID */
  productCategoryId: number
  /** Product promotion name */
  promotionName: string
  /** Product promotion breakdown amount */
  promotionAmount: number
  /** Coupon discount breakdown amount */
  couponAmount: number
  /** Points discount breakdown amount */
  integrationAmount: number
  /** Item amount after discounts */
  realAmount: number
  /** Bonus points */
  giftIntegration: number
  /** Bonus growth value */
  giftGrowth: number
  /** Product sales attributes */
  productAttr: string
}

/** Order operation history */
export type OmsOrderOperateHistory = {
  /** History record ID */
  id: number
  /** Order ID */
  orderId: number
  /** Operator: user; system; backend admin */
  operateMan: string
  /** Operation time */
  createTime: string
  /** Order status: 0 -> pending payment; 1 -> pending shipment; 2 -> shipped; 3 -> completed; 4 -> closed; 5 -> invalid */
  orderStatus: number
  /** Remark */
  note: string
}

/** Order list query params */
export type OrderQueryParam = PageParam & {
  /** Order number */
  orderSn?: string
  /** Recipient name / phone */
  receiverKeyword?: string
  /** Order status: 0 -> pending payment; 1 -> pending shipment; 2 -> shipped; 3 -> completed; 4 -> closed; 5 -> invalid */
  status?: number
  /** Order type: 0 -> normal order; 1 -> flash sale order */
  orderType?: number
  /** Order source: 0 -> PC order; 1 -> app order */
  sourceType?: number
  /** Order submission time */
  createTime?: string
}

/** Order shipment params */
export type OmsOrderDeliveryParam = {
  /** Order ID */
  orderId: number
  /** Shipping company */
  deliveryCompany?: string
  /** Tracking number */
  deliverySn?: string
}

/** Order detail info */
export type OmsOrderDetail = OmsOrder & {
  /** Order product list */
  orderItemList: OmsOrderItem[]
  /** Order operation records */
  historyList: OmsOrderOperateHistory[]
}

/** Order recipient info params */
export type OmsReceiverInfoParam = {
  /** Order ID */
  orderId: number
  /** Recipient name */
  receiverName?: string
  /** Recipient phone */
  receiverPhone?: string
  /** Recipient postal code */
  receiverPostCode?: string
  /** Detailed address */
  receiverDetailAddress?: string
  /** Province / municipality */
  receiverProvince?: string
  /** City */
  receiverCity?: string
  /** District */
  receiverRegion?: string
  /** Order status: 0 -> pending payment; 1 -> pending shipment; 2 -> shipped; 3 -> completed; 4 -> closed; 5 -> invalid */
  status: number
}

/** Order fee info params */
export type OmsMoneyInfoParam = {
  /** Order ID */
  orderId: number
  /** Shipping fee amount */
  freightAmount: number
  /** Discount amount adjusted by admin in backend */
  discountAmount: number
  /** Order status: 0 -> pending payment; 1 -> pending shipment; 2 -> shipped; 3 -> completed; 4 -> closed; 5 -> invalid */
  status: number
}
