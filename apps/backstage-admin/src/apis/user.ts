import type { CommonPage } from '@/types/common'
import http from '@/utils/http'

// ==================== Admin API for User Management ====================

export interface User {
  id: number
  username: string
  nickname: string
  avatar: string
  email: string
  telephone: string
  password: string
  typeId: number
  integration: number
  growth: number
  memberLevelId: number
  status: number
  createTime: string
  updateTime: string
}

export interface UserQueryParams {
  pageNum?: number
  pageSize?: number
  keyword?: string
  typeId?: number
  email?: string
}

export interface UpdateUserParams {
  nickname?: string
  avatar?: string
  email?: string
  telephone?: string
  username?: string
  integration?: number
  growth?: number
  memberLevelId?: number
}

export interface UserStatistics {
  totalUsers: number
  activeUsers: number
  vipUsers: number
}

// Get user list (admin)
export function getUserListAPI(params: UserQueryParams) {
  return http<CommonPage<User>>({
    url: '/api/admin/user/list',
    method: 'GET',
    params
  })
}

// Get user by ID
export function getUserByIdAPI(id: number) {
  return http<User>({
    url: `/api/admin/user/${id}`,
    method: 'GET'
  })
}

// Update user
export function updateUserAPI(id: number, data: UpdateUserParams) {
  return http({
    url: `/api/admin/user/${id}`,
    method: 'PUT',
    data
  })
}

// Update user status
export function updateUserStatusAPI(id: number, status: number) {
  return http({
    url: `/api/admin/user/${id}/status`,
    method: 'PUT',
    params: { status }
  })
}

// Delete user
export function deleteUserAPI(id: number) {
  return http({
    url: `/api/admin/user/${id}`,
    method: 'DELETE'
  })
}

// Batch delete users
export function deleteUsersBatchAPI(ids: string) {
  return http({
    url: '/api/admin/user/batch',
    method: 'DELETE',
    params: { ids }
  })
}

// Get user statistics
export function getUserStatisticsAPI() {
  return http<UserStatistics>({
    url: '/api/admin/user/statistics',
    method: 'GET'
  })
}