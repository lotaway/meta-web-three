/** Subject entity */
export type CmsSubject = {
  /** ID */
  id?: number
  /** Category ID */
  categoryId?: number
  /** Title */
  title?: string
  /** Subject main image */
  pic?: string
  /** Associated product count */
  productCount?: number
  /** Recommendation status */
  recommendStatus?: number
  /** Creation time */
  createTime?: string
  /** Favorite count */
  collectCount?: number
  /** View count */
  readCount?: number
  /** Review count */
  commentCount?: number
  /** Gallery images, comma-separated */
  albumPics?: string
  /** Description */
  description?: string
  /** Display status: 0 -> hidden; 1 -> shown */
  showStatus?: number
  /** Share count */
  forwardCount?: number
  /** Subject category name */
  categoryName?: string
  /** Content */
  content?: string
}
