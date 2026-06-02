import type { CommonPage } from '@/types/common'
import http from '@/utils/http'

// ==================== Admin API for Group Buying Management ====================

export interface GroupBuyActivity {
  id: number
  activityName: string
  productId: number
  productName: string
  singlePrice: number
  groupPrice: number
  requiredQuantity: number
  currentQuantity: number
  status: number
  startTime: string
  endTime: string
  validityHours: number
  createdAt: string
  updatedAt: string
}

export interface GroupBuyTeam {
  id: number
  activityId: number
  teamNo: string
  leaderId: number
  requiredQuantity: number
  currentQuantity: number
  status: string
  orderId: number
  expireTime: string
  createdAt: string
  updatedAt: string
}

export interface GroupBuyOrder {
  id: number
  teamId: number
  activityId: number
  userId: number
  orderNo: string
  orderId: number
  productId: number
  quantity: number
  unitPrice: number
  totalAmount: number
  status: string
  isLeader: boolean
  createdAt: string
  updatedAt: string
}

export interface GroupBuyActivityQueryParams {
  pageNum?: number
  pageSize?: number
  activityName?: string
  productId?: number
  status?: number
}

export interface CreateGroupBuyActivityParams {
  activityName: string
  productId: number
  productName: string
  singlePrice: number
  groupPrice: number
  requiredQuantity: number
  validityHours: number
  startTime: string
  endTime: string
  status?: number
}

export interface GroupBuyTeamQueryParams {
  pageNum?: number
  pageSize?: number
  activityId?: number
  leaderId?: number
  status?: string
}

export interface GroupBuyOrderQueryParams {
  pageNum?: number
  pageSize?: number
  teamId?: number
  activityId?: number
  userId?: number
  status?: string
}

export interface GroupBuyStatistics {
  totalActivities: number
  activeActivities: number
  totalTeams: number
  successTeams: number
  totalOrders: number
}

// Get group buy activity list (admin)
export function getGroupBuyActivityListAPI(params: GroupBuyActivityQueryParams) {
  return http<CommonPage<GroupBuyActivity>>({
    url: '/api/admin/group-buy/activity/list',
    method: 'get',
    params,
  })
}

// Get group buy activity by ID (admin)
export function getGroupBuyActivityByIdAPI(id: number) {
  return http<GroupBuyActivity>({
    url: `/api/admin/group-buy/activity/${id}`,
    method: 'get',
  })
}

// Create group buy activity (admin)
export function createGroupBuyActivityAPI(data: CreateGroupBuyActivityParams) {
  return http<GroupBuyActivity>({
    url: '/api/admin/group-buy/activity',
    method: 'post',
    data,
  })
}

// Update group buy activity (admin)
export function updateGroupBuyActivityAPI(id: number, data: Partial<CreateGroupBuyActivityParams>) {
  return http({
    url: `/api/admin/group-buy/activity/${id}`,
    method: 'put',
    data,
  })
}

// Delete group buy activity (admin)
export function deleteGroupBuyActivityAPI(id: number) {
  return http({
    url: `/api/admin/group-buy/activity/${id}`,
    method: 'delete',
  })
}

// Get group buy team list (admin)
export function getGroupBuyTeamListAPI(params: GroupBuyTeamQueryParams) {
  return http<CommonPage<GroupBuyTeam>>({
    url: '/api/admin/group-buy/team/list',
    method: 'get',
    params,
  })
}

// Get group buy team by ID (admin)
export function getGroupBuyTeamByIdAPI(id: number) {
  return http<GroupBuyTeam>({
    url: `/api/admin/group-buy/team/${id}`,
    method: 'get',
  })
}

// Get group buy order list (admin)
export function getGroupBuyOrderListAPI(params: GroupBuyOrderQueryParams) {
  return http<CommonPage<GroupBuyOrder>>({
    url: '/api/admin/group-buy/order/list',
    method: 'get',
    params,
  })
}

// Get group buy statistics (admin)
export function getGroupBuyStatisticsAPI() {
  return http<GroupBuyStatistics>({
    url: '/api/admin/group-buy/statistics',
    method: 'get',
  })
}