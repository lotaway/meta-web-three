/** Menu */
export type UmsMenu = {
  /** Unique ID */
  id?: number
  /** Parent ID */
  parentId: number
  /** Creation time */
  createTime?: string
  /** Menu name */
  title: string
  /** Menu level */
  level?: number
  /** Menu sort order */
  sort: number
  /** Frontend name */
  name: string
  /** Frontend icon */
  icon: string
  /** Hidden on frontend */
  hidden: number
}

/** Backend menu node wrapper */
export type UmsMenuNode = UmsMenu & {
  /** Child menu */
  children: UmsMenu[]
}
