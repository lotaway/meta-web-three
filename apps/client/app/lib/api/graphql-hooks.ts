import { apiClient } from './client'
import {
  type RecommendationsQuery,
  type RecommendationsQueryVariables,
  type RecommendationsBySceneQuery,
  type RecommendationsBySceneQueryVariables,
  type RecommendationMetricsQuery,
  type RecommendationMetricsQueryVariables,
  type UserBehaviorHistoryQuery,
  type UserBehaviorHistoryQueryVariables,
  type GenerateRecommendationMutation,
  type GenerateRecommendationMutationVariables,
  type RecordBehaviorMutation,
  type RecordBehaviorMutationVariables,
  type MarkRecommendationClickedMutation,
  type MarkRecommendationClickedMutationVariables,
  type MarkRecommendationPurchasedMutation,
  type MarkRecommendationPurchasedMutationVariables,
} from '@/src/generated/graphql/types'

interface GraphQLResponse<T> {
  data?: T
  errors?: Array<{ message: string }>
}

async function query<TData, TVariables>(
  queryStr: string,
  variables: TVariables,
): Promise<TData> {
  const res = await apiClient.post<GraphQLResponse<TData>>('/graphql', { query: queryStr, variables })
  if (res.errors) throw new Error(res.errors[0].message)
  return res.data as TData
}

async function mutate<TData, TVariables>(
  mutationStr: string,
  variables: TVariables,
): Promise<TData> {
  return query<TData, TVariables>(mutationStr, variables)
}

const GET_RECOMMENDATIONS = `query Recommendations($userId: ID!, $limit: Int) {
  recommendations(userId: $userId, limit: $limit) {
    id userId productId score algorithm reason position isClicked isPurchased createdAt
  }
}`

const GET_RECOMMENDATIONS_BY_SCENE = `query RecommendationsByScene($userId: ID!, $scene: String!, $limit: Int) {
  recommendationsByScene(userId: $userId, scene: $scene, limit: $limit) {
    id userId productId score algorithm reason position isClicked isPurchased createdAt
  }
}`

const GET_RECOMMENDATION_METRICS = `query RecommendationMetrics($userId: ID!) {
  recommendationMetrics(userId: $userId) {
    totalRecommendations clickedCount purchasedCount clickThroughRate conversionRate
  }
}`

const GET_BEHAVIOR_HISTORY = `query UserBehaviorHistory($userId: ID!, $limit: Int) {
  userBehaviorHistory(userId: $userId, limit: $limit) {
    id userId productId behaviorType behaviorValue timestamp sessionId source
  }
}`

const MUTATE_GENERATE = `mutation GenerateRecommendation($userId: ID!, $scene: String!, $algorithm: String!, $maxItems: Int) {
  generateRecommendation(userId: $userId, scene: $scene, algorithm: $algorithm, maxItems: $maxItems) {
    id userId scene algorithm score status items { skuCode skuName score rank reason }
    clickCount conversionCount impressionCount createdAt expiresAt
  }
}`

const MUTATE_RECORD_BEHAVIOR = `mutation RecordBehavior($userId: ID!, $skuCode: String!, $behaviorType: String!) {
  recordBehavior(userId: $userId, skuCode: $skuCode, behaviorType: $behaviorType)
}`

const MUTATE_CLICK = `mutation MarkRecommendationClicked($id: ID!) {
  markRecommendationClicked(id: $id)
}`

const MUTATE_PURCHASE = `mutation MarkRecommendationPurchased($id: ID!) {
  markRecommendationPurchased(id: $id)
}`

export const recommendationHooks = {
  getRecommendations: (userId: number, limit = 10) =>
    query<RecommendationsQuery, RecommendationsQueryVariables>(GET_RECOMMENDATIONS, { userId: String(userId), limit }),

  getRecommendationsByScene: (userId: number, scene: string, limit = 10) =>
    query<RecommendationsBySceneQuery, RecommendationsBySceneQueryVariables>(
      GET_RECOMMENDATIONS_BY_SCENE, { userId: String(userId), scene, limit }),

  getMetrics: (userId: number) =>
    query<RecommendationMetricsQuery, RecommendationMetricsQueryVariables>(
      GET_RECOMMENDATION_METRICS, { userId: String(userId) }),

  getBehaviorHistory: (userId: number, limit = 50) =>
    query<UserBehaviorHistoryQuery, UserBehaviorHistoryQueryVariables>(
      GET_BEHAVIOR_HISTORY, { userId: String(userId), limit }),

  generateRecommendation: (userId: number, scene: string, algorithm: string, maxItems = 10) =>
    mutate<GenerateRecommendationMutation, GenerateRecommendationMutationVariables>(
      MUTATE_GENERATE, { userId: String(userId), scene, algorithm, maxItems }),

  recordBehavior: (userId: number, skuCode: string, behaviorType: string) =>
    mutate<RecordBehaviorMutation, RecordBehaviorMutationVariables>(
      MUTATE_RECORD_BEHAVIOR, { userId: String(userId), skuCode, behaviorType }),

  markClicked: (id: number) =>
    mutate<MarkRecommendationClickedMutation, MarkRecommendationClickedMutationVariables>(
      MUTATE_CLICK, { id: String(id) }),

  markPurchased: (id: number) =>
    mutate<MarkRecommendationPurchasedMutation, MarkRecommendationPurchasedMutationVariables>(
      MUTATE_PURCHASE, { id: String(id) }),
}
