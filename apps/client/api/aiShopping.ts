import { getToken } from '@/lib/api/interceptors'

// ============================================================
// AI shopping client (hand-written, styled after src/generated/api)
// Routing: gateway discovery locator + ZK dispatch, URL starts with the service id
//   {gateway}/recommendation-service/api/ai-shopping/...
// Responses use ApiResponse { code, message, data }
// ============================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_BACK_API_HOST ?? 'http://localhost:10081'

export type CorrectionSource = 'LLM' | 'LOCAL' | 'NONE'

export interface TextCorrection {
  original: string
  corrected: string
  changed: boolean
  suggestions: string[]
  source: CorrectionSource
}

export interface AiProductMatch {
  productId: number
  name: string
  pic?: string
  price?: string
  score: number
  reason: string
}

export interface AiSearchResult {
  query: string
  correction: TextCorrection
  matches: AiProductMatch[]
}

interface ApiEnvelope<T> {
  code: string
  message: string
  data: T
}

const AI_SHOPPING_BASE = '/recommendation-service/api/ai-shopping'

function authHeaders(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function jsonHeaders(): Record<string, string> {
  return {
    'Content-Type': 'application/json',
    ...authHeaders(),
  }
}

async function requestEnvelope<T>(
  url: string,
  init: RequestInit
): Promise<ApiEnvelope<T>> {
  const response = await fetch(`${API_BASE_URL}${url}`, init)
  if (!response.ok) {
    throw new Error(`AI shopping request failed: ${response.status}`)
  }
  return (await response.json()) as ApiEnvelope<T>
}

export const aiShoppingApi = {
  /** Text correction */
  async correctText(text: string): Promise<TextCorrection> {
    const envelope = await requestEnvelope<TextCorrection>(
      `${AI_SHOPPING_BASE}/text-correct`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ text }),
      }
    )
    return envelope.data
  },

  /** Smart match */
  async smartMatch(query: string, topK?: number): Promise<AiProductMatch[]> {
    const envelope = await requestEnvelope<AiProductMatch[]>(
      `${AI_SHOPPING_BASE}/smart-match`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ query, topK }),
      }
    )
    return envelope.data
  },

  /** Image search (multipart image upload) */
  async imageSearch(image: Blob, topK?: number): Promise<AiProductMatch[]> {
    const form = new FormData()
    form.append('image', image)
    if (topK) {
      form.append('topK', String(topK))
    }
    const envelope = await requestEnvelope<AiProductMatch[]>(
      `${AI_SHOPPING_BASE}/image-search`,
      {
        method: 'POST',
        headers: authHeaders(),
        body: form,
      }
    )
    return envelope.data
  },

  /** One-stop search: correction then smart match */
  async search(
    query: string,
    topK?: number,
    userId?: number
  ): Promise<AiSearchResult> {
    const envelope = await requestEnvelope<AiSearchResult>(
      `${AI_SHOPPING_BASE}/search`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ query, topK, userId }),
      }
    )
    return envelope.data
  },
}
