/** Generic response result wrapper */
export type CommonResult<T> = {
  /** Status code (backend uses a string, e.g. "0000") */
  code: number | string
  /** Message */
  message: string
  /** Wrapped data */
  data: T
}

/** Generic paginated result wrapper */
export type CommonPage<T> = {
  /** Current page number */
  pageNum: number
  /** Page size */
  pageSize: number
  /** Total pages */
  totalPage: number
  /** Total count */
  total: number
  /** Paginated data */
  list: T[]
}

/** Generic pagination request params */
export type PageParam = {
  /** Current page number, starting from 1 */
  pageNum: number
  /** Page size, default 10 */
  pageSize: number
  /** Search keyword */
  keyword?: string
}

/** el-cascader data structure */
export type ElCascaderDataVo = {
  /** Display label */
  label: string
  /** Corresponding ID */
  value: number
  /** Child data */
  children?: ElCascaderDataVo[]
}

/** el-select data structure */
export type ElSelectDataVo = {
  /** Display label */
  label: string
  /** Corresponding value */
  value: string | number
}

/** el-transfer data structure */
export type ElTransferDataVo = {
  /** Binding key */
  key?: number
  /** Display label */
  label?: string
  /** Whether available */
  disabled?: boolean
}
