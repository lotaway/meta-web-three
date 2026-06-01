import http from '@/utils/http'

export interface Review {
  id?: number
  orderId?: number
  orderItemId?: number
  productId?: number
  skuId?: number
  userId?: number
  userNickname?: string
  userAvatar?: string
  storeId?: number
  storeName?: string
  rating?: number
  content?: string
  images?: string
  status?: number
  statusDesc?: string
  likeCount?: number
  replyCount?: number
  replyContent?: string
  createTime?: string
  updateTime?: string
}

export interface ReviewQueryParam {
  pageNum?: number
  pageSize?: number
  productId?: number
  storeId?: number
  userId?: number
  status?: number
}

export interface ReviewReplyParam {
  reviewId: number
  content: string
}

export function getReviewListAPI(params: ReviewQueryParam) {
  return http<Review[]>({
    url: '/api/review/list',
    method: 'get',
    params: params,
  })
}

export function getReviewByIdAPI(id: number) {
  return http<Review>({
    url: '/api/review/' + id,
    method: 'get',
  })
}

export function getReviewsByProductAPI(productId: number, sortBy?: string) {
  return http<Review[]>({
    url: '/api/review/product/' + productId,
    method: 'get',
    params: { sortBy },
  })
}

export function getReviewsByStoreAPI(storeId: number) {
  return http<Review[]>({
    url: '/api/review/store/' + storeId,
    method: 'get',
  })
}

export function getReviewsByUserAPI(userId: number) {
  return http<Review[]>({
    url: '/api/review/user/' + userId,
    method: 'get',
  })
}

export function approveReviewAPI(id: number) {
  return http<Review>({
    url: '/api/review/' + id + '/approve',
    method: 'post',
  })
}

export function rejectReviewAPI(id: number) {
  return http<Review>({
    url: '/api/review/' + id + '/reject',
    method: 'post',
  })
}

export function replyReviewAPI(data: ReviewReplyParam) {
  return http<Review>({
    url: '/api/review/reply',
    method: 'post',
    data: data,
  })
}

export function likeReviewAPI(id: number) {
  return http<Review>({
    url: '/api/review/' + id + '/like',
    method: 'post',
  })
}