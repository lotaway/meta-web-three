import type { PageParam } from './common'

/** Home advertisement entity */
export type SmsHomeAdvertise = {
  /** ID */
  id?: number
  /** Advertisement name */
  name?: string
  /** Carousel position: 0 -> PC home carousel; 1 -> app home carousel */
  type?: number
  /** Image */
  pic?: string
  /** Start time */
  startTime?: string
  /** End time */
  endTime?: string
  /** Online status: 0 -> offline; 1 -> online */
  status?: number
  /** Click count */
  clickCount?: number
  /** Order count */
  orderCount?: number
  /** Link URL */
  url?: string
  /** Remark */
  note?: string
  /** Sort order */
  sort?: number
}

/** Home advertisement list query params */
export type HomeAdvertiseQueryParam = PageParam & {
  /** Advertisement name */
  name?: string
  /** Carousel position: 0 -> PC home carousel; 1 -> app home carousel */
  type?: number
  /** End time */
  endTime?: string
}
