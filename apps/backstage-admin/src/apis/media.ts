import http from '@/utils/http'

export interface UserStorage {
  id: number
  userId: number
  totalUsed: number
  createdAt: string
  updatedAt: string
}

export interface UserStorageQueryParam {
  pageNum?: number
  pageSize?: number
  userId?: number
}

export interface MediaStatistics {
  totalUsers: number
  totalUsed: number
  maxUsed: number
  maxUsedUserId: number | null
  averageUsed: number
}

export interface QuotaConfig {
  maxFileSize: number
  totalQuota: number
}

export interface QuotaConfigResponse {
  quotas: Record<string, QuotaConfig>
}

export function getUserStorageListAPI(params: UserStorageQueryParam) {
  return http<{ list: UserStorage[]; total: number; pageNum: number; pageSize: number }>({
    url: '/api/admin/media/storage/list',
    method: 'get',
    params,
  })
}

export function getUserStorageByIdAPI(id: number) {
  return http<UserStorage>({ url: `/api/admin/media/storage/${id}`, method: 'get' })
}

export function getUserStorageByUserIdAPI(userId: number) {
  return http<UserStorage>({ url: `/api/admin/media/storage/user/${userId}`, method: 'get' })
}

export function deleteUserStorageAPI(id: number) {
  return http<void>({ url: `/api/admin/media/storage/${id}`, method: 'delete' })
}

export function getMediaStatisticsAPI() {
  return http<MediaStatistics>({ url: '/api/admin/media/statistics', method: 'get' })
}

export function getQuotaConfigAPI() {
  return http<QuotaConfigResponse>({ url: '/api/admin/media/quota/config', method: 'get' })
}