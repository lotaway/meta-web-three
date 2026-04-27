export interface OrderSignaturePayload {
  buyerId: string
  sku: string
  quantity: number
  price: string
  discount: string
  freight: string
  requestKey: string
}

export interface SignedOrderResult {
  total: string
  signature: string
}

export type OrderSignatureStatus = 'idle' | 'loading' | 'success' | 'error'
