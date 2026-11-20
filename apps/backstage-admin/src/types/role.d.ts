/** Role info */
export type UmsRole = {
  /** ID */
  id?: number
  /** Name */
  name: string
  /** Description */
  description?: string
  /** Admin user count */
  adminCount?: number
  /** Creation time */
  createTime?: string
  /** Enabled status: 0 -> disabled; 1 -> enabled */
  status: number
  /** Sort order */
  sort?: number
}
