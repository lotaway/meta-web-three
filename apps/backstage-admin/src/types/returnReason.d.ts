/** Return reason */
export type OmsOrderReturnReason = {
  /** ID */
  id?: number
  /** Return type */
  name: string
  /** Sort order */
  sort: number
  /** Status: 0 -> disabled; 1 -> enabled */
  status: number
  /** Added time */
  createTime?: string
}
