import http from '@/utils/http'

export interface LiveRoom {
  id?: number
  anchorId: number
  anchorName?: string
  roomName: string
  coverImage?: string
  description?: string
  status: 'PENDING' | 'LIVE' | 'ENDED' | 'CANCELLED'
  viewerCount: number
  startTime?: string
  endTime?: string
  createdAt?: string
  updatedAt?: string
}

export interface Anchor {
  id?: number
  userId: number
  anchorName: string
  avatar?: string
  description?: string
  followerCount: number
  createdAt?: string
  updatedAt?: string
}

export interface LiveProduct {
  id?: number
  roomId: number
  productId: number
  productName?: string
  price: number
  discountPrice: number
  stock: number
  soldCount: number
  createdAt?: string
  updatedAt?: string
}

export interface LiveComment {
  id?: number
  roomId: number
  userId: number
  userName: string
  content: string
  createdAt?: string
}

export interface LiveOrder {
  id?: number
  roomId: number
  productId: number
  productName?: string
  userId: number
  userName?: string
  quantity: number
  totalAmount: number
  status: 'PENDING' | 'PAID' | 'SHIPPED' | 'COMPLETED' | 'CANCELLED'
  createdAt?: string
  updatedAt?: string
}

export interface LiveQueryParam {
  pageNum?: number
  pageSize?: number
  status?: string
  anchorId?: number
  roomName?: string
  startTime?: string
  endTime?: string
}

export interface AnchorQueryParam {
  pageNum?: number
  pageSize?: number
  anchorName?: string
  userId?: number
}

// Live Room APIs
export function getLiveRoomListAPI(params: LiveQueryParam) {
  return http<{ data: LiveRoom[]; total: number }>({
    url: '/api/live/room',
    method: 'get',
    params: params,
  })
}

export function getLiveRoomByIdAPI(id: number) {
  return http<LiveRoom>({
    url: '/api/live/room/' + id,
    method: 'get',
  })
}

export function getLiveRoomsByAnchorAPI(anchorId: number) {
  return http<LiveRoom[]>({
    url: '/api/live/room/anchor/' + anchorId,
    method: 'get',
  })
}

export function startLiveRoomAPI(data: {
  anchorId: number
  roomName: string
  coverImage?: string
  description?: string
}) {
  return http<LiveRoom>({
    url: '/api/live/room/start',
    method: 'post',
    data: data,
  })
}

export function endLiveRoomAPI(roomId: number) {
  return http<LiveRoom>({
    url: '/api/live/room/end/' + roomId,
    method: 'post',
  })
}

// Anchor APIs
export function getAnchorListAPI(params: AnchorQueryParam) {
  return http<{ data: Anchor[]; total: number }>({
    url: '/api/live/anchor',
    method: 'get',
    params: params,
  })
}

export function getAnchorByIdAPI(id: number) {
  return http<Anchor>({
    url: '/api/live/anchor/' + id,
    method: 'get',
  })
}

export function getAnchorByUserIdAPI(userId: number) {
  return http<Anchor>({
    url: '/api/live/anchor/user/' + userId,
    method: 'get',
  })
}

export function createAnchorAPI(data: {
  userId: number
  anchorName: string
  avatar?: string
  description?: string
}) {
  return http<Anchor>({
    url: '/api/live/anchor/create',
    method: 'post',
    data: data,
  })
}

// Live Product APIs
export function getLiveProductsByRoomAPI(roomId: number) {
  return http<LiveProduct[]>({
    url: '/api/live/room/' + roomId + '/products',
    method: 'get',
  })
}

export function attachProductAPI(data: {
  roomId: string
  productId: string
  price: string
  discountPrice: string
  stock: string
}) {
  return http<LiveProduct>({
    url: '/api/live/room/product/attach',
    method: 'post',
    data: data,
  })
}

// Live Comment APIs
export function getLiveCommentsByRoomAPI(roomId: number) {
  return http<LiveComment[]>({
    url: '/api/live/room/' + roomId + '/comments',
    method: 'get',
  })
}

export function postCommentAPI(data: {
  roomId: number
  userId: number
  userName: string
  content: string
}) {
  return http<LiveComment>({
    url: '/api/live/room/comment',
    method: 'post',
    data: data,
  })
}

// Live Order APIs
export function getLiveOrdersByRoomAPI(roomId: number) {
  return http<LiveOrder[]>({
    url: '/api/live/room/' + roomId + '/orders',
    method: 'get',
  })
}

export function getLiveOrdersByUserAPI(userId: number) {
  return http<LiveOrder[]>({
    url: '/api/live/room/user/' + userId + '/orders',
    method: 'get',
  })
}

export function createLiveOrderAPI(data: {
  roomId: number
  productId: number
  userId: number
  quantity: number
}) {
  return http<LiveOrder>({
    url: '/api/live/room/order/create',
    method: 'post',
    data: data,
  })
}