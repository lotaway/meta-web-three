import type { RouteRecordRaw } from 'vue-router'

/** vue-router route extension */
export type RouteRecordExt = RouteRecordRaw & {
  /** Hidden on frontend */
  hidden?: boolean
  /** Frontend sort order */
  sort?: number
  /** Child route */
  children?: RouteRecordExt[]
  /** Whether always shown */
  alwaysShow?: boolean
}
