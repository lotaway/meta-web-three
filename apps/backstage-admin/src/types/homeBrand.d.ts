/** Home brand */
export type SmsHomeBrand = {
  /** ID */
  id?: number
  /** Brand ID */
  brandId: number
  /** Brand name */
  brandName: string
  /** Recommendation status: 0 -> not recommended; 1 -> recommending */
  recommendStatus?: number
  /** Sort order */
  sort?: number
}

/** Home brand list query params */
export type HomeBrandQueryParam = PageParam & {
  /** Brand name */
  brandName?: string
  /** Recommendation status: 0 -> not recommended; 1 -> recommending */
  recommendStatus?: number
}
