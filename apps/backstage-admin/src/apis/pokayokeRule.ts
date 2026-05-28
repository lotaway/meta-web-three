import http from '@/utils/http'

export type RuleStatus = 'DRAFT' | 'ACTIVE' | 'INACTIVE'
export type RuleType = 'MATERIAL_CHECK' | 'SEQUENCE_CHECK' | 'PARAMETER_CHECK' | 'STATION_CHECK'

export interface PokayokeRule {
  id?: number
  ruleCode: string
  ruleName: string
  ruleType: RuleType
  status: RuleStatus
  priority?: number
  workstationId?: string
  conditionExpression?: string
  actionType?: string
  actionConfig?: string
  errorMessage?: string
  createdAt?: string
  updatedAt?: string
}

export interface CreateRuleRequest {
  ruleCode: string
  ruleName: string
  ruleType: RuleType
  priority?: number
  workstationId?: string
  conditionExpression?: string
  actionType?: string
  actionConfig?: string
  errorMessage?: string
}

export interface UpdateRuleRequest {
  ruleName?: string
  ruleType?: RuleType
  priority?: number
  workstationId?: string
  conditionExpression?: string
  actionType?: string
  actionConfig?: string
  errorMessage?: string
}

export function createRuleAPI(data: CreateRuleRequest) {
  return http<PokayokeRule>({
    url: '/api/mes/pokayoke/rules',
    method: 'post',
    data,
  })
}

export function updateRuleAPI(id: number, data: UpdateRuleRequest) {
  return http<PokayokeRule>({
    url: `/api/mes/pokayoke/rules/${id}`,
    method: 'put',
    data,
  })
}

export function deleteRuleAPI(id: number) {
  return http({
    url: `/api/mes/pokayoke/rules/${id}`,
    method: 'delete',
  })
}

export function getRuleByIdAPI(id: number) {
  return http<PokayokeRule>({
    url: `/api/mes/pokayoke/rules/${id}`,
    method: 'get',
  })
}

export function getRuleListAPI(params?: {
  status?: RuleStatus
  ruleType?: RuleType
}) {
  return http<PokayokeRule[]>({
    url: '/api/mes/pokayoke/rules',
    method: 'get',
    params,
  })
}

export function activateRuleAPI(id: number) {
  return http({
    url: `/api/mes/pokayoke/rules/${id}/activate`,
    method: 'post',
  })
}

export function deactivateRuleAPI(id: number) {
  return http({
    url: `/api/mes/pokayoke/rules/${id}/deactivate`,
    method: 'post',
  })
}

export function getActiveRulesAPI() {
  return http<PokayokeRule[]>({
    url: '/api/mes/pokayoke/rules/active',
    method: 'get',
  })
}

export function getRulesByWorkstationAPI(workstationId: string) {
  return http<PokayokeRule[]>({
    url: `/api/mes/pokayoke/rules/workstation/${workstationId}`,
    method: 'get',
  })
}