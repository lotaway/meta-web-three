/** Product category info */
export type PmsProductCategory = {
  /** ID */
  id?: number
  /** Parent category number: 0 means top-level category */
  parentId: number
  /** Category name */
  name: string
  /** Category level: 0 -> level 1; 1 -> level 2 */
  level?: number
  /** Product quantity */
  productCount?: number
  /** Product unit */
  productUnit?: string
  /** Whether shown in nav bar: 0 -> hidden; 1 -> shown */
  navStatus: number
  /** Display status: 0 -> hidden; 1 -> shown */
  showStatus: number
  /** Sort order */
  sort?: number
  /** Icon */
  icon?: string
  /** Keyword */
  keywords?: string
  /** Description */
  description?: string
  /** Filter attribute IDs, added later */
  productAttributeIdList?: number[]
}

/** Product category extension */
export type PmsProductCategoryExt = PmsProductCategory & {
  /** Child category */
  children?: PmsProductCategory[]
}
