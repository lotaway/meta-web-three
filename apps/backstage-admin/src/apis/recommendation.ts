import type { CommonPage } from '@/types/common'
import type { Recommendation, RecommendationRule } from '@/types/recommendation'
import http from '@/utils/http'

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