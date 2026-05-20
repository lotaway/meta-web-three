import { API_BASE_URL } from '@/api/generated'

export interface FlashOrderRequest {
  sessionId: number
  productId: number
  skuId: number
  productName: string
  productPic: string
  quantity: number
  flashPrice: number
  orderRemark?: string
}

export interface FlashOrderResponse {
  orderId: number
}

export async function createFlashOrder(token: string, request: FlashOrderRequest): Promise<FlashOrderResponse> {
  const res = await fetch(`${API_BASE_URL}/promotion-service/flash/order`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(request),
  })
  const data = await res.json()
  if (data.code !== '0000') {
    throw new Error(data.message || '闪购下单失败')
  }
  return { orderId: data.data?.orderId }
}
