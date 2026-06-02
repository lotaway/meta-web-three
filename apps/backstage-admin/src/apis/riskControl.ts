import http from '@/utils/http'
import type { CommonPage } from '@/types/common'

// ==================== Risk Event Types ====================

export interface RiskEvent {
  id?: number
  eventId?: string
  userId?: string
  scene?: string
  eventType?: string
  riskScore?: number
  riskLevel?: string
  decision?: string
  description?: string
  details?: string
  status?: number
  createTime?: number
  updateTime?: number
}

export interface RiskRule {
  id?: number
  ruleId?: string
  ruleName?: string
  scene?: string
  ruleType?: string
  condition?: string
  score?: number
  riskLevel?: string
  priority?: number
  status?: number
  createTime?: number
  updateTime?: number
}

export interface RiskEventQueryParams {
  pageNum?: number
  pageSize?: number
  userId?: string
  scene?: string
  riskLevel?: string
  decision?: string
  status?: number
}

export interface RiskRuleQueryParams {
  pageNum?: number
  pageSize?: number
  ruleName?: string
  scene?: string
  ruleType?: string
  status?: number
}

export interface RiskStatistics {
  total: number
  high: number
  medium: number
  low: number
  reviewPending: number
}

// ==================== Risk Event API ====================

// Get risk event list
export function getRiskEventListAPI(params: RiskEventQueryParams) {
  return http<CommonPage<RiskEvent>>({
    url: '/api/admin/risk/event/list',
    method: 'get',
    params,
  })
}

// Get risk event by ID
export function getRiskEventByIdAPI(id: number) {
  return http<RiskEvent>({
    url: `/api/admin/risk/event/${id}`,
    method: 'get',
  })
}

// Update risk event status
export function updateRiskEventStatusAPI(id: number, status: number) {
  return http({
    url: `/api/admin/risk/event/${id}/status`,
    method: 'put',
    params: { status },
  })
}

// Get risk event statistics
export function getRiskStatisticsAPI() {
  return http<RiskStatistics>({
    url: '/api/admin/risk/event/statistics',
    method: 'get',
  })
}

// ==================== Risk Rule API ====================

// Get risk rule list
export function getRiskRuleListAPI(params: RiskRuleQueryParams) {
  return http<CommonPage<RiskRule>>({
    url: '/api/admin/risk/rule/list',
    method: 'get',
    params,
  })
}

// Get risk rule by ID
export function getRiskRuleByIdAPI(id: number) {
  return http<RiskRule>({
    url: `/api/admin/risk/rule/${id}`,
    method: 'get',
  })
}

// Create risk rule
export function createRiskRuleAPI(data: RiskRule) {
  return http<RiskRule>({
    url: '/api/admin/risk/rule',
    method: 'post',
    data,
  })
}

// Update risk rule
export function updateRiskRuleAPI(id: number, data: RiskRule) {
  return http({
    url: `/api/admin/risk/rule/${id}`,
    method: 'put',
    data,
  })
}

// Delete risk rule
export function deleteRiskRuleAPI(id: number) {
  return http({
    url: `/api/admin/risk/rule/${id}`,
    method: 'delete',
  })
}

// Update risk rule status
export function updateRiskRuleStatusAPI(id: number, status: number) {
  return http({
    url: `/api/admin/risk/rule/${id}/status`,
    method: 'put',
    params: { status },
  })
}