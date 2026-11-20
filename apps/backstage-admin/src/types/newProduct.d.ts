import type { PageParam } from './common'

/** Home new-product recommendation entity */
export type SmsHomeNewProduct = {
  /** ID */
  id?: number
  /** Product ID */
  productId: number
  /** Product name */
  productName: string
  /** Recommendation status: 0 -> not recommended; 1 -> recommending */
  recommendStatus?: number
  /** Sort order */
  sort?: number
}

/** Home new-product recommendation list query params */
export type NewProductQueryParam = PageParam & {
  /** Product name */
  productName?: string
  /** Recommendation status: 0 -> not recommended; 1 -> recommending */
  recommendStatus?: number
}
