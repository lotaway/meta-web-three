/** Member level info */
export type UmsMemberLevel = {
  /** ID */
  id?: number
  /** Level name */
  name: string
  /** Growth points */
  growthPoint: number
  /** Whether default level: 0 -> no; 1 -> yes */
  defaultStatus: number
  /** Free shipping threshold */
  freeFreightPoint: number
  /** Growth value per review */
  commentGrowthPoint: number
  /** Whether free-shipping privilege */
  priviledgeFreeFreight: number
  /** Whether check-in privilege */
  priviledgeSignIn: number
  /** Whether review-reward privilege */
  priviledgeComment: number
  /** Whether exclusive-event privilege */
  priviledgePromotion: number
  /** Whether member-price privilege */
  priviledgeMemberPrice: number
  /** Whether birthday privilege */
  priviledgeBirthday: number
  /** Remark */
  note: string
}
