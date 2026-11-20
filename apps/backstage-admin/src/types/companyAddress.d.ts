/** Company shipping/receiving address */
export type OmsCompanyAddress = {
  /** ID */
  id?: number
  /** Address name */
  addressName: string
  /** Default shipping address: 0 -> no; 1 -> yes */
  sendStatus: number
  /** Whether default receiving address: 0 -> no; 1 -> yes */
  receiveStatus: number
  /** Sender/recipient name */
  name: string
  /** Recipient phone */
  phone: string
  /** Province / municipality */
  province: string
  /** City */
  city: string
  /** District */
  region: string
  /** Detailed address */
  detailAddress: string
}
