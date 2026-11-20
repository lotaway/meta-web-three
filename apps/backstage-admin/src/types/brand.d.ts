/** Brand info */
export type PmsBrand = {
  /** ID */
  id?: number
  /** Brand name */
  name: string
  /** Initial letter */
  firstLetter: string
  /** Sort order */
  sort: number
  /** Whether brand manufacturer: 0 -> no; 1 -> yes */
  factoryStatus: number
  /** Display status */
  showStatus: number
  /** Product count */
  productCount?: number
  /** Product review count */
  productCommentCount?: number
  /** Brand logo */
  logo: string
  /** Zone banner image */
  bigPic: string
  /** Brand story */
  brandStory: string
}
