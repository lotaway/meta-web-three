import type { CommonResult } from '@/types/common'
import http from '@/utils/http'

// ==================== AI Shopping Admin API ====================

export interface AiShoppingConfig {
  configKey: string
  configValue: string
  description?: string
  updatedAt?: string
}

export type AiSearchType = 'TEXT_CORRECT' | 'SMART_MATCH' | 'IMAGE_SEARCH' | 'COMBINED_SEARCH'

export interface AiSearchLog {
  id?: number
  userId?: number
  searchType: AiSearchType
  queryText?: string
  correctedText?: string
  resultCount?: number
  responseTimeMs?: number
  createdAt?: string
}

export interface AiIndexStatus {
  status?: Record<string, unknown>
  vectorStore?: string
  collectionText?: string
  collectionImage?: string
}

// List runtime config overrides
export function listAiShoppingConfigsAPI() {
  return http<AiShoppingConfig[]>({
    url: '/api/admin/ai-shopping/config',
    method: 'get',
  })
}

// Save runtime config override
export function saveAiShoppingConfigAPI(data: Partial<AiShoppingConfig>) {
  return http<AiShoppingConfig>({
    url: '/api/admin/ai-shopping/config',
    method: 'post',
    data,
  })
}

// Delete runtime config override
export function deleteAiShoppingConfigAPI(key: string) {
  return http<null>({
    url: `/api/admin/ai-shopping/config/${encodeURIComponent(key)}`,
    method: 'delete',
  })
}

// Rebuild vector index (type: all | text | image)
export function rebuildAiShoppingIndexAPI(type: string = 'all') {
  return http<{ started: boolean; type: string }>({
    url: '/api/admin/ai-shopping/index/rebuild',
    method: 'post',
    params: { type },
  })
}

// Get vector index status
export function getAiShoppingIndexStatusAPI() {
  return http<AiIndexStatus>({
    url: '/api/admin/ai-shopping/index/status',
    method: 'get',
  })
}

// Test AI provider connectivity (type: embedding | image | llm)
export function testAiProviderAPI(type: string = 'embedding') {
  return http<{ success: boolean; type: string; responseTimeMs?: number; error?: string }>({
    url: '/api/admin/ai-shopping/provider/test',
    method: 'post',
    params: { type },
  })
}

// Recent AI search logs
export function getAiShoppingLogsAPI(limit: number = 50) {
  return http<AiSearchLog[]>({
    url: '/api/admin/ai-shopping/logs',
    method: 'get',
    params: { limit },
  })
}

export type { CommonResult }
