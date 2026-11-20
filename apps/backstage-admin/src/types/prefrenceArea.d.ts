/** Featured zone info */
export type CmsPrefrenceArea = {
  /** ID */
  id?: number
  /** Name */
  name: string
  /** Subtitle */
  subTitle: string
  /** Sort order */
  sort: number
  /** Display status: 0 -> hidden; 1 -> shown */
  showStatus: number
  /** Display image */
  pic?: string
}
