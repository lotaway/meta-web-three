import http from '@/utils/http'

// Cart Item Types
export interface CartItem {
  id?: number
  productId: number
  productSkuId: number
  memberId: number
  quantity: number
  price: number
  productPic: string
  productName: string
  productSubTitle: string
  productSkuCode: string
  memberNickname: string
  createDate: string
  modifyDate: string
  deleteStatus: number
  productCategoryId: number
  productBrand: string
  productSn: string
  productAttr: string
  promotionTag: string
  promotionType: string
  discountAmount: number
  promotionStartTime: string
  promotionEndTime: string
}

export interface CartItemQueryParam {
  pageNum?: number
  pageSize?: number
  memberId?: number
  memberNickname?: string
}

export interface CartItemAddParam {
  productId: number
  productSkuId: number
  quantity: number
}

export interface CartItemUpdateAttrParam {
  productSkuId: number
  quantity: number
  productAttr: string
}

// Cart APIs
export function getCartListAPI(memberId?: number) {
  return http<CartItem[]>({
    url: '/api/cart/list',
    method: 'get',
    headers: memberId ? { 'X-User-Id': String(memberId) } : {}
  })
}

export function getCartListWithPromotionAPI(memberId?: number) {
  return http<CartItem[]>({
    url: '/api/cart/list/promotion',
    method: 'get',
    headers: memberId ? { 'X-User-Id': String(memberId) } : {}
  })
}

export function addCartItemAPI(data: CartItemAddParam, memberId?: number) {
  return http<number>({
    url: '/api/cart/add',
    method: 'post',
    data: data,
    headers: memberId ? { 'X-User-Id': String(memberId) } : {}
  })
}

export function updateCartQuantityAPI(id: number, quantity: number, memberId?: number) {
  return http<number>({
    url: '/api/cart/update/quantity',
    method: 'get',
    params: { id, quantity },
    headers: memberId ? { 'X-User-Id': String(memberId) } : {}
  })
}

export function deleteCartItemsAPI(ids: number[], memberId?: number) {
  return http<number>({
    url: '/api/cart/delete',
    method: 'post',
    params: { ids: ids.join(',') },
    headers: memberId ? { 'X-User-Id': String(memberId) } : {}
  })
}

export function clearCartAPI(memberId?: number) {
  return http<number>({
    url: '/api/cart/clear',
    method: 'post',
    headers: memberId ? { 'X-User-Id': String(memberId) } : {}
  })
}

export function updateCartAttributesAPI(id: number, data: CartItemUpdateAttrParam, memberId?: number) {
  return http<void>({
    url: '/api/cart/update/attr',
    method: 'post',
    params: { id },
    data: data,
    headers: memberId ? { 'X-User-Id': String(memberId) } : {}
  })
}

export function getProductOptionsAPI(productId: number, memberId?: number) {
  return http<CartItem>({
    url: `/api/cart/getProduct/${productId}`,
    method: 'get',
    headers: memberId ? { 'X-User-Id': String(memberId) } : {}
  })
}