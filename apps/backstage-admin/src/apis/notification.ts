import http from '@/utils/http'

export interface Notification {
  id: number
  userId: number
  userName?: string
  title: string
  content: string
  type: string
  typeDesc?: string
  channel: string
  channelDesc?: string
  priority: number
  priorityDesc?: string
  status: string
  statusDesc?: string
  sendTime?: string
  readTime?: string
  readStatus: boolean
  createTime: string
  remark?: string
}

export interface NotificationSendDTO {
  userId: number
  title: string
  content: string
  type: string
  channel: string
  priority?: number
}

export interface TemplateConfigDTO {
  templateCode: string
  templateName: string
  content: string
  params: Record<string, string>
  channel: string
  type: string
}

// Send notification
export function sendNotificationAPI(data: NotificationSendDTO) {
  return http<Notification>({ url: '/api/notification/send', method: 'post', data })
}

// Send notification using template
export function sendWithTemplateAPI(templateCode: string, userId: number, params: Record<string, string>) {
  return http<Notification>({
    url: '/api/notification/send-template',
    method: 'post',
    params: { templateCode, userId },
    data: params,
  })
}

// Mark notification as read
export function markAsReadAPI(id: number) {
  return http<Notification>({ url: `/api/notification/${id}/read`, method: 'post' })
}

// Get notification by ID
export function getNotificationByIdAPI(id: number) {
  return http<Notification>({ url: `/api/notification/${id}`, method: 'get' })
}

// Get all notifications for a user
export function getUserNotificationsAPI(userId: number) {
  return http<Notification[]>({ url: `/api/notification/user/${userId}`, method: 'get' })
}

// Get unread notifications for a user
export function getUnreadNotificationsAPI(userId: number) {
  return http<Notification[]>({ url: `/api/notification/user/${userId}/unread`, method: 'get' })
}

// Get all notifications (admin)
export function getAllNotificationsAPI(params?: { page?: number; size?: number; type?: string; status?: string }) {
  return http<{ list: Notification[]; total: number }>({ url: '/api/notification/list', method: 'get', params })
}

// Delete notification
export function deleteNotificationAPI(id: number) {
  return http({ url: `/api/notification/${id}`, method: 'delete' })
}

// Batch delete notifications
export function batchDeleteNotificationsAPI(ids: number[]) {
  return http({ url: '/api/notification/batch-delete', method: 'post', data: ids })
}

// Mark all as read for a user
export function markAllAsReadAPI(userId: number) {
  return http({ url: `/api/notification/user/${userId}/read-all`, method: 'post' })
}

// Get notification templates
export function getTemplatesAPI() {
  return http<TemplateConfigDTO[]>({ url: '/api/notification/templates', method: 'get' })
}

// Create/update template
export function saveTemplateAPI(data: TemplateConfigDTO) {
  return http<TemplateConfigDTO>({ url: '/api/notification/template', method: 'post', data })
}

// Delete template
export function deleteTemplateAPI(templateCode: string) {
  return http({ url: `/api/notification/template/${templateCode}`, method: 'delete' })
}

// Get notification statistics
export function getNotificationStatsAPI(userId: number) {
  return http<{ total: number; unread: number; read: number }>({ url: `/api/notification/user/${userId}/stats`, method: 'get' })
}
