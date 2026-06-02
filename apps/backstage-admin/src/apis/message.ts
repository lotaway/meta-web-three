import http from '@/utils/http'

export interface Notification {
  id: number
  userId: number
  title: string
  content: string
  icon?: string
  imageUrl?: string
  type: string
  relatedId?: string
  readStatus: number
  createTime: string
}

export interface NotificationQueryParam {
  pageNum?: number
  pageSize?: number
  userId?: number
  title?: string
  type?: string
  readStatus?: number
  startDate?: string
  endDate?: string
}

export interface NotificationStatistics {
  total: number
  unread: number
  read: number
}

export function getNotificationListAPI(params: NotificationQueryParam) {
  return http<{ list: Notification[]; total: number; pageNum: number; pageSize: number }>({
    url: '/api/admin/notification/list',
    method: 'get',
    params,
  })
}

export function getNotificationByIdAPI(id: number) {
  return http<Notification>({ url: `/api/admin/notification/${id}`, method: 'get' })
}

export function createNotificationAPI(data: {
  userId?: number
  title: string
  content: string
  type: string
  relatedId?: string
  icon?: string
  imageUrl?: string
}) {
  return http<Notification>({ url: '/api/admin/notification/create', method: 'post', data })
}

export function batchCreateNotificationAPI(data: {
  userIds: number[]
  title: string
  content: string
  type: string
  relatedId?: string
  icon?: string
  imageUrl?: string
}) {
  return http<void>({ url: '/api/admin/notification/batch-create', method: 'post', data })
}

export function deleteNotificationAPI(id: number) {
  return http<void>({ url: `/api/admin/notification/${id}`, method: 'delete' })
}

export function batchDeleteNotificationAPI(ids: number[]) {
  return http<void>({ url: '/api/admin/notification/batch-delete', method: 'delete', data: ids })
}

export function getNotificationStatisticsAPI() {
  return http<NotificationStatistics>({ url: '/api/admin/notification/statistics', method: 'get' })
}

export function exportNotificationAPI(params: {
  userId?: number
  title?: string
  type?: string
  readStatus?: number
}) {
  return http<Notification[]>({ url: '/api/admin/notification/export', method: 'get', params })
}