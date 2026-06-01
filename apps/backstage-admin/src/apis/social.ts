import http from '@/utils/http'

// Share Reward Config
export interface ShareRewardConfig {
  id?: number
  configName: string
  rewardType: number
  fixedAmount?: number
  percentage?: number
  maxRewardCount?: number
  maxRewardAmount?: number
  status: number
  validFrom?: string
  validTo?: string
  createdAt?: string
  updatedAt?: string
}

// Share Record
export interface ShareRecord {
  id?: number
  sharerId: number
  sharerName?: string
  sharedItemId: number
  itemType: string
  shareChannel: string
  shareUrl: string
  clickCount: number
  purchaseCount: number
  rewardAmount: number
  status: string
  createdAt?: string
  updatedAt?: string
}

// Distribution Relation
export interface DistributionRelation {
  id?: number
  userId: number
  userName?: string
  referrerId: number
  referrerName?: string
  level: number
  rootReferrerId: number
  status: string
  bindTime?: string
  createdAt?: string
}

// Distribution Reward
export interface DistributionReward {
  id?: number
  referrerId: number
  referrerName?: string
  buyerId: number
  buyerName?: string
  orderId: number
  orderAmount: number
  commissionAmount: number
  level: number
  status: string
  settledTime?: string
  createdAt?: string
}

// Community
export interface Community {
  id?: number
  communityName: string
  description?: string
  ownerId: number
  ownerName?: string
  avatarUrl?: string
  memberCount: number
  maxMembers: number
  status: string
  inviteCode: string
  createdAt?: string
  updatedAt?: string
}

// Community Member
export interface CommunityMember {
  id?: number
  communityId: number
  communityName?: string
  userId: number
  userName?: string
  nickname: string
  role: string
  messageCount: number
  status: string
  joinedAt?: string
  lastActiveAt?: string
}

// Group Buy Activity
export interface GroupBuyActivity {
  id?: number
  activityName: string
  productId: number
  productName?: string
  groupPrice: number
  originalPrice: number
  requiredCount: number
  currentCount: number
  startTime?: string
  endTime?: string
  status: 'PENDING' | 'ACTIVE' | 'ENDED' | 'CANCELLED'
  createdAt?: string
  updatedAt?: string
}

// Group Buy Order
export interface GroupBuyOrder {
  id?: number
  activityId: number
  activityName?: string
  userId: number
  userName?: string
  productId: number
  productName?: string
  quantity: number
  totalAmount: number
  status: 'PENDING' | 'SUCCESS' | 'FAILED' | 'REFUNDED'
  createdAt?: string
  updatedAt?: string
}

// Query Params
export interface ShareRewardConfigQueryParam {
  pageNum?: number
  pageSize?: number
  configName?: string
  status?: number
}

export interface ShareRecordQueryParam {
  pageNum?: number
  pageSize?: number
  sharerId?: number
  itemType?: string
  status?: string
}

export interface DistributionRelationQueryParam {
  pageNum?: number
  pageSize?: number
  referrerId?: number
  level?: number
  status?: string
}

export interface CommunityQueryParam {
  pageNum?: number
  pageSize?: number
  ownerId?: number
  status?: string
  communityName?: string
}

export interface GroupBuyActivityQueryParam {
  pageNum?: number
  pageSize?: number
  productId?: number
  status?: string
  activityName?: string
}

// Share Reward Config APIs
export function getShareRewardConfigListAPI(params: ShareRewardConfigQueryParam) {
  return http<{ data: ShareRewardConfig[]; total: number }>({
    url: '/api/social-commerce/share-reward-config',
    method: 'get',
    params: params,
  })
}

export function createShareRewardConfigAPI(data: {
  name: string
  rewardType: number
  fixedAmount?: number
  percentage?: number
  maxRewardCount?: number
  maxRewardAmount?: number
  validFrom?: string
  validTo?: string
}) {
  return http<ShareRewardConfig>({
    url: '/api/social-commerce/share-reward-config/create',
    method: 'post',
    data: data,
  })
}

// Share Record APIs
export function getShareRecordListAPI(params: ShareRecordQueryParam) {
  return http<{ data: ShareRecord[]; total: number }>({
    url: '/api/social-commerce/share-record',
    method: 'get',
    params: params,
  })
}

export function createShareRecordAPI(data: {
  sharerId: number
  itemId: number
  itemType: string
  shareChannel: string
}) {
  return http<{ shareUrl: string }>({
    url: '/api/social-commerce/share-record/create',
    method: 'post',
    data: data,
  })
}

// Distribution Relation APIs
export function getDistributionRelationListAPI(params: DistributionRelationQueryParam) {
  return http<{ data: DistributionRelation[]; total: number }>({
    url: '/api/social-commerce/distribution-relation',
    method: 'get',
    params: params,
  })
}

export function bindDistributionRelationAPI(data: {
  userId: number
  referrerId: number
}) {
  return http<{ success: boolean }>({
    url: '/api/social-commerce/distribution-relation/bind',
    method: 'post',
    data: data,
  })
}

// Distribution Reward APIs
export function getDistributionRewardListAPI(params: {
  pageNum?: number
  pageSize?: number
  referrerId?: number
  status?: string
}) {
  return http<{ data: DistributionReward[]; total: number }>({
    url: '/api/social-commerce/distribution-reward',
    method: 'get',
    params: params,
  })
}

// Community APIs
export function getCommunityListAPI(params: CommunityQueryParam) {
  return http<{ data: Community[]; total: number }>({
    url: '/api/social-commerce/community',
    method: 'get',
    params: params,
  })
}

export function createCommunityAPI(data: {
  name: string
  description?: string
  ownerId: number
  avatarUrl?: string
  maxMembers?: number
}) {
  return http<Community>({
    url: '/api/social-commerce/community/create',
    method: 'post',
    data: data,
  })
}

// Group Buy Activity APIs
export function getGroupBuyActivityListAPI(params: GroupBuyActivityQueryParam) {
  return http<{ data: GroupBuyActivity[]; total: number }>({
    url: '/api/group-buying/activity',
    method: 'get',
    params: params,
  })
}

export function createGroupBuyActivityAPI(data: {
  activityName: string
  productId: number
  groupPrice: number
  originalPrice: number
  requiredCount: number
  startTime?: string
  endTime?: string
}) {
  return http<GroupBuyActivity>({
    url: '/api/group-buying/activity/create',
    method: 'post',
    data: data,
  })
}

// Group Buy Order APIs
export function getGroupBuyOrderListAPI(params: {
  pageNum?: number
  pageSize?: number
  activityId?: number
  userId?: number
  status?: string
}) {
  return http<{ data: GroupBuyOrder[]; total: number }>({
    url: '/api/group-buying/order',
    method: 'get',
    params: params,
  })
}