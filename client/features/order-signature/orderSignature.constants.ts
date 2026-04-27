import type { OrderSignaturePayload } from '@/features/order-signature/orderSignature.types'

export const ORDER_SIGNATURE_DEMO_PAYLOAD: OrderSignaturePayload = {
  buyerId: 'U12345',
  sku: 'META001',
  quantity: 1,
  price: '299.00',
  discount: '0.00',
  freight: '15.00',
  requestKey: 'demo-key',
}
