import type { PageParam } from './common'

/**
 * Home subject recommendation
 */
export type SmsHomeRecommendSubject = {
  /** ID */
  id?: number
  /** Subject ID */
  subjectId: number
  /** Subject name */
  subjectName: string
  /** Recommendation status: 0 -> not recommended; 1 -> recommending */
  recommendStatus?: number
  /** Sort order */
  sort?: number
}

/** Home subject list query params */
export type HomeSubjectQueryParam = PageParam & {
  /** Brand name */
  subjectName?: string
  /** Recommendation status: 0 -> not recommended; 1 -> recommending */
  recommendStatus?: number
}
