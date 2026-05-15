import { useState, useCallback, useEffect } from 'react'
import { API_BASE_URL, DEFAULT_USER_ID } from '@/api/generated'

export interface Conversation {
  id: string
  sessionId: string
  customerId: number
  agentId: number | null
  status: string
  channel: string
  productId: number | null
  orderId: number | null
  createTime: string
  satisfactionScore: number | null
}

export interface CsMessage {
  id: string
  sessionId: string
  messageId: string
  senderType: string
  senderId: number
  msgType: string
  content: string
  timestamp: string
  readStatus: boolean
}

export function useConversation() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(false)

  const userId = DEFAULT_USER_ID

  const fetchMyConversations = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE_URL}/cs/conversation/my`, {
        headers: { 'X-User-Id': String(userId) },
      })
      const json = await res.json()
      if (json.data) setConversations(json.data)
    } catch { }
    setLoading(false)
  }, [userId])

  const createConversation = useCallback(async (
    channel: string, productId?: number, orderId?: number,
  ): Promise<Conversation | null> => {
    try {
      const res = await fetch(`${API_BASE_URL}/cs/conversation/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User-Id': String(userId) },
        body: JSON.stringify({ channel, productId, orderId }),
      })
      const json = await res.json()
      if (json.data) {
        setConversations((prev) => [json.data, ...prev])
        return json.data
      }
      return null
    } catch { return null }
  }, [userId])

  const fetchMessages = useCallback(async (sessionId: string): Promise<CsMessage[]> => {
    try {
      const res = await fetch(`${API_BASE_URL}/cs/message/list?sessionId=${sessionId}`)
      const json = await res.json()
      return json.data || []
    } catch { return [] }
  }, [])

  const sendMessage = useCallback(async (
    sessionId: string, msgType: string, content: string,
  ): Promise<CsMessage | null> => {
    try {
      const res = await fetch(`${API_BASE_URL}/cs/message/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId, senderType: 'CUSTOMER', senderId: userId,
          msgType, content,
        }),
      })
      const json = await res.json()
      return json.data || null
    } catch { return null }
  }, [userId])

  const closeConversation = useCallback(async (sessionId: string) => {
    try {
      await fetch(`${API_BASE_URL}/cs/conversation/close?sessionId=${sessionId}`, {
        method: 'POST',
      })
      setConversations((prev) => prev.filter((c) => c.sessionId !== sessionId))
    } catch { }
  }, [])

  const rateConversation = useCallback(async (sessionId: string, score: number) => {
    try {
      await fetch(`${API_BASE_URL}/cs/conversation/rate?sessionId=${sessionId}&score=${score}`, {
        method: 'POST',
      })
    } catch { }
  }, [])

  useEffect(() => { fetchMyConversations() }, [fetchMyConversations])

  return {
    conversations, loading, fetchMyConversations, createConversation,
    fetchMessages, sendMessage, closeConversation, rateConversation,
  }
}
