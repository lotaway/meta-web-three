export interface Recommendation {
  id: number
  userId: number
  scene: string
  algorithm: string
  items: RecommendationItem[]
  createdAt: string
  updatedAt: string
}

export interface RecommendationItem {
  skuCode: string
  score: number
  rank: number
}

export interface RecommendationRule {
  id: number
  ruleName: string
  scene: string
  type: string
  isActive: boolean
  createdAt: string
  updatedAt: string
}

export interface RecommendationQueryParam {
  userId?: number
  scene?: string
  algorithm?: string
  page?: number
  pageSize?: number
}

export interface RecommendationRuleQueryParam {
  scene?: string
  type?: string
  page?: number
  pageSize?: number
}