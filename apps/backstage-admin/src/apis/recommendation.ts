import type { CommonPage } from '@/types/common'
import type { Recommendation, RecommendationRule } from '@/types/recommendation'
import http from '@/utils/http'

// ==================== Admin API for Recommendation Management ====================

export interface RecommendationRuleQueryParams {
  pageNum?: number
  pageSize?: number
  ruleName?: string
  scene?: string
  status?: string
}

export interface CreateRecommendationRuleParams {
  ruleName: string
  scene: string
  type: string
  priority?: number
  maxItems?: number
  conditions?: string
  exclusions?: string
  boostFactor?: number
}

export interface RecommendationQueryParams {
  pageNum?: number
  pageSize?: number
  userId?: number
  scene?: string
}

export interface RecommendationStatistics {
  totalRules: number
  activeRules: number
  totalRecommendations: number
  totalClicks: number
  totalConversions: number
  avgClickThroughRate: number
  avgConversionRate: number
  sceneDistribution: Record<string, number>
  algorithmDistribution: Record<string, number>
}

// Get recommendation rule list (admin)
export function getRecommendationRuleListAPI(params: RecommendationRuleQueryParams) {
  return http<CommonPage<RecommendationRule>>({
    url: '/api/admin/recommendation/rule/list',
    method: 'get',
    params,
  })
}

// Get recommendation rule by ID (admin)
export function getRecommendationRuleByIdAPI(id: number) {
  return http<RecommendationRule>({
    url: `/api/admin/recommendation/rule/${id}`,
    method: 'get',
  })
}

// Create recommendation rule (admin)
export function createRecommendationRuleAPI(data: CreateRecommendationRuleParams) {
  return http<RecommendationRule>({
    url: '/api/admin/recommendation/rule',
    method: 'post',
    data,
  })
}

// Update recommendation rule (admin)
export function updateRecommendationRuleAPI(id: number, data: Partial<CreateRecommendationRuleParams>) {
  return http({
    url: `/api/admin/recommendation/rule/${id}`,
    method: 'put',
    data,
  })
}

// Delete recommendation rule (admin)
export function deleteRecommendationRuleAPI(id: number) {
  return http({
    url: `/api/admin/recommendation/rule/${id}`,
    method: 'delete',
  })
}

// Activate recommendation rule
export function activateRecommendationRuleAPI(id: number) {
  return http({
    url: `/api/admin/recommendation/rule/${id}/activate`,
    method: 'put',
  })
}

// Pause recommendation rule
export function pauseRecommendationRuleAPI(id: number) {
  return http({
    url: `/api/admin/recommendation/rule/${id}/pause`,
    method: 'put',
  })
}

// Archive recommendation rule
export function archiveRecommendationRuleAPI(id: number) {
  return http({
    url: `/api/admin/recommendation/rule/${id}/archive`,
    method: 'put',
  })
}

// Get recommendation records (admin)
export function getRecommendationListAPI(params: RecommendationQueryParams) {
  return http<CommonPage<Recommendation>>({
    url: '/api/admin/recommendation/list',
    method: 'get',
    params,
  })
}

// Get recommendation statistics
export function getRecommendationStatisticsAPI() {
  return http<RecommendationStatistics>({
    url: '/api/admin/recommendation/statistics',
    method: 'get',
  })
}

// Re-export types for external usage
export type { Recommendation, RecommendationRule }

export interface RecommendationQueryParams {
  userId?: number
  scene?: string
  page?: number
  pageSize?: number
}

export interface RecommendationRuleQueryParams {
  scene?: string
  page?: number
  pageSize?: number
}

export interface GenerateRecommendationParams {
  userId: number
  scene: string
  algorithm: string
  maxItems?: number
}

export interface CreateRuleParams {
  ruleName: string
  scene: string
  type: string
}

export interface RecordBehaviorParams {
  userId: number
  skuCode: string
  behaviorType: string
}

// Get recommendation by ID
export function getRecommendationByIdAPI(id: number) {
  return http<Recommendation>({
    url: `/api/recommendation/${id}`,
    method: 'get',
  })
}

// Get user recommendations
export function getUserRecommendationsAPI(userId: number, params?: { scene?: string }) {
  return http<Recommendation[]>({
    url: `/api/recommendation/user/${userId}`,
    method: 'get',
    params,
  })
}

// Generate recommendation
export function generateRecommendationAPI(data: GenerateRecommendationParams) {
  return http<{ recommendationId: number }>({
    url: '/api/recommendation/generate',
    method: 'post',
    data,
  })
}

// Record user behavior
export function recordBehaviorAPI(data: RecordBehaviorParams) {
  return http({
    url: '/api/recommendation/behavior',
    method: 'post',
    data,
  })
}

// Create recommendation rule
export function createRuleAPI(data: CreateRuleParams) {
  return http<{ ruleId: number }>({
    url: '/api/recommendation/rule',
    method: 'post',
    data,
  })
}

// Activate rule
export function activateRuleAPI(id: number) {
  return http({
    url: `/api/recommendation/rule/${id}/activate`,
    method: 'post',
  })
}

// Delete rule
export function deleteRuleAPI(id: number) {
  return http({
    url: `/api/recommendation/rule/${id}`,
    method: 'delete',
  })
}

// Get rules by scene
export function getRulesBySceneAPI(scene: string) {
  return http<RecommendationRule[]>({
    url: `/api/recommendation/rule/scene/${scene}`,
    method: 'get',
  })
}