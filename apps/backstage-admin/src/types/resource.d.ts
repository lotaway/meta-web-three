import type { PageParam } from './common'

/** Resource info */
export type UmsResource = {
  /** ID */
  id?: number
  /** Creation time */
  createTime?: string
  /** Resource name */
  name: string
  /** Resource URL */
  url: string
  /** Description */
  description?: string
  /** Resource category ID */
  categoryId: number
  /** Whether selected, used on frontend */
  checked?: boolean
}

/** Resource category info */
export type UmsResourceCategory = {
  /** ID */
  id?: number
  /** Creation time */
  createTime?: string
  /** Category name */
  name: string
  /** Sort order */
  sort: number
  /** Whether selected, used on frontend */
  checked?: boolean
}

/** Resource list query params */
export type ResourceQueryParam = PageParam & {
  /** Resource name keyword */
  nameKeyword?: string
  /** Resource path keyword */
  urlKeyword?: string
  /** Resource category ID */
  categoryId?: number
}
