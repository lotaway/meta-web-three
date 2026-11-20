import type { PageParam } from './common'
import type { OmsCompanyAddress } from './companyAddress'

/** Order return application */
export type OmsOrderReturnApply = {
  /** Primary key ID */
  id: number
  /** Order ID */
  orderId: number
  /** Shipping address ID */
  companyAddressId: number
  /** Returned product ID */
  productId: number
  /** Order number */
  orderSn: string
  /** Application time */
  createTime: string
  /** Member username */
  memberUsername: string
  /** Refund amount */
  returnAmount: number
  /** Return person name */
  returnName: string
  /** Return person phone */
  returnPhone: string
  /** Application status: 0 -> pending; 1 -> returning; 2 -> completed; 3 -> rejected */
  status: number
  /** Processing time */
  handleTime: string
  /** Product image */
  productPic: string
  /** Product name */
  productName: string
  /** Product brand */
  productBrand: string
  /** Product sales attributes: color: red; size: xl; */
  productAttr: string
  /** Return quantity */
  productCount: number
  /** Product unit price */
  productPrice: number
  /** Product actual paid unit price */
  productRealPrice: number
  /** Reason */
  reason: string
  /** Description */
  description: string
  /** Proof images, comma-separated */
  proofPics: string
  /** Processing remark */
  handleNote: string
  /** Handler */
  handleMan: string
  /** Recipient */
  receiveMan: string
  /** Receipt time */
  receiveTime: string
  /** Receiving remark */
  receiveNote: string
}

/** Return application query params */
export type ReturnApplyQueryParam = PageParam & {
  /** Service order number */
  id?: number
  /** Recipient name / phone */
  receiverKeyword?: string
  /** Application status: 0 -> pending; 1 -> returning; 2 -> completed; 3 -> rejected */
  status?: number
  /** Application time */
  createTime?: string
  /** Handler */
  handleMan?: string
  /** Processing time */
  handleTime?: string
}

/** Order return application result wrapper */
export type OmsOrderReturnApplyResult = OmsOrderReturnApply & {
  /** Company receiving address */
  companyAddress: OmsCompanyAddress
}

/** Return application status update params */
export type OmsUpdateStatusParam = {
  /** Service order number */
  id: number
  /** Associated shipping address ID */
  companyAddressId: number
  /** Confirmed refund amount */
  returnAmount: number
  /** Processing remark */
  handleNote: string
  /** Handler */
  handleMan: string
  /** Receiving remark */
  receiveNote: string
  /** Recipient */
  receiveMan: string
  /** Application status: 1 -> returning; 2 -> completed; 3 -> rejected */
  status: number
}
