/** SKU stock info */
export type PmsSkuStock = {
  /** ID */
  id?: number
  /** Product ID */
  productId?: number
  /** SKU code */
  skuCode?: string
  /** Price */
  price?: number
  /** Stock */
  stock?: number
  /** Warning stock level */
  lowStock?: number
  /** Display image */
  pic?: string
  /** Sales volume */
  sale?: number
  /** Single-item promotion price */
  promotionPrice?: number
  /** Locked stock */
  lockStock?: number
  /** Product sales attributes, JSON format */
  spData?: string
}
