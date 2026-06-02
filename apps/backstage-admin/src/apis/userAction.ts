import http from '@/utils/http'

export interface ProductCollection {
  id: number
  userId: number
  productId: number
  productName: string
  productPic: string
  createTime: string
}

export interface ReadHistory {
  id: number
  userId: number
  productId: number
  productName: string
  productPic: string
  createTime: string
}

export interface BrandAttention {
  id: number
  userId: number
  brandId: number
  brandName: string
  brandLogo: string
  createTime: string
}

export interface ProductComment {
  id: number
  productId: number
  userId: number
  memberNickName: string
  productName: string
  star: number
  content: string
  pics: string
  productAttribute: string
  showStatus: number
  collectCount: number
  readCount: number
  replayCount: number
  createTime: string
}

export interface UserActionQueryParam {
  pageNum?: number
  pageSize?: number
  userId?: number
  productId?: number
  productName?: string
  brandId?: number
  brandName?: string
  showStatus?: number
  star?: number
}

export interface UserActionStatistics {
  totalCollections: number
  totalHistories: number
  totalAttentions: number
  totalComments: number
  visibleComments: number
}

export function getCollectionListAPI(params: UserActionQueryParam) {
  return http<{ list: ProductCollection[]; total: number; pageNum: number; pageSize: number }>({
    url: '/api/admin/user-action/collection/list',
    method: 'get',
    params,
  })
}

export function getCollectionByIdAPI(id: number) {
  return http<ProductCollection>({ url: `/api/admin/user-action/collection/${id}`, method: 'get' })
}

export function deleteCollectionAPI(id: number) {
  return http<void>({ url: `/api/admin/user-action/collection/${id}`, method: 'delete' })
}

export function batchDeleteCollectionsAPI(ids: number[]) {
  return http<void>({ url: '/api/admin/user-action/collection/batch', method: 'delete', data: ids })
}

export function getHistoryListAPI(params: UserActionQueryParam) {
  return http<{ list: ReadHistory[]; total: number; pageNum: number; pageSize: number }>({
    url: '/api/admin/user-action/history/list',
    method: 'get',
    params,
  })
}

export function deleteHistoryAPI(id: number) {
  return http<void>({ url: `/api/admin/user-action/history/${id}`, method: 'delete' })
}

export function batchDeleteHistoriesAPI(ids: number[]) {
  return http<void>({ url: '/api/admin/user-action/history/batch', method: 'delete', data: ids })
}

export function getAttentionListAPI(params: UserActionQueryParam) {
  return http<{ list: BrandAttention[]; total: number; pageNum: number; pageSize: number }>({
    url: '/api/admin/user-action/attention/list',
    method: 'get',
    params,
  })
}

export function deleteAttentionAPI(id: number) {
  return http<void>({ url: `/api/admin/user-action/attention/${id}`, method: 'delete' })
}

export function batchDeleteAttentionsAPI(ids: number[]) {
  return http<void>({ url: '/api/admin/user-action/attention/batch', method: 'delete', data: ids })
}

export function getCommentListAPI(params: UserActionQueryParam) {
  return http<{ list: ProductComment[]; total: number; pageNum: number; pageSize: number }>({
    url: '/api/admin/user-action/comment/list',
    method: 'get',
    params,
  })
}

export function getCommentByIdAPI(id: number) {
  return http<ProductComment>({ url: `/api/admin/user-action/comment/${id}`, method: 'get' })
}

export function updateCommentStatusAPI(id: number, showStatus: number) {
  return http<void>({
    url: `/api/admin/user-action/comment/${id}/status`,
    method: 'put',
    params: { showStatus },
  })
}

export function deleteCommentAPI(id: number) {
  return http<void>({ url: `/api/admin/user-action/comment/${id}`, method: 'delete' })
}

export function batchDeleteCommentsAPI(ids: number[]) {
  return http<void>({ url: '/api/admin/user-action/comment/batch', method: 'delete', data: ids })
}

export function getUserActionStatisticsAPI() {
  return http<UserActionStatistics>({ url: '/api/admin/user-action/statistics', method: 'get' })
}