/** Order settings info */
export type OmsOrderSetting = {
  /** ID */
  id: number
  /** Flash sale order timeout (minutes) */
  flashOrderOvertime?: number
  /** Normal order timeout (minutes) */
  normalOrderOvertime?: number
  /** Auto-confirm after shipment (days) */
  confirmOvertime?: number
  /** Auto-complete transaction, no after-sale allowed (days) */
  finishOvertime?: number
  /** Auto-positive review after completion (days) */
  commentOvertime?: number
}
