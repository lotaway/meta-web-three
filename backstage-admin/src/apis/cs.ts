import http from '@/utils/http'
import type { CommonResult } from '@/types/common'

export interface Conversation {
  id: string
  sessionId: string
  customerId: number
  agentId: number | null
  status: string
  channel: string
  productId: number | null
  orderId: number | null
  queuePosition: number | null
  createTime: string
  activeTime: string
  endTime: string | null
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

export interface Agent {
  id: number
  adminId: number
  nickname: string
  avatar: string
  status: string
  maxConcurrent: number
  currentLoad: number
  groupId: number | null
  createTime: string
}

export interface QuickReply {
  id: number | null
  groupId: number | null
  title: string
  content: string
  msgType: string
  sort: number
  createTime: string
}

export function getQueuingConversationsAPI() {
  return http<Conversation[]>({ url: '/cs/conversation/queuing', method: 'get' })
}

export function getAgentConversationsAPI(agentId: number) {
  return http<Conversation[]>({ url: '/cs/conversation/agent', method: 'get', params: { agentId } })
}

export function closeConversationAPI(sessionId: string) {
  return http({ url: '/cs/conversation/close', method: 'post', params: { sessionId } })
}

export function getMessagesAPI(sessionId: string) {
  return http<CsMessage[]>({ url: '/cs/message/list', method: 'get', params: { sessionId } })
}

export function sendMessageAPI(sessionId: string, senderType: string, senderId: number, msgType: string, content: string) {
  return http<CsMessage>({
    url: '/cs/message/send',
    method: 'post',
    data: { sessionId, senderType, senderId, msgType, content },
  })
}

export function agentOnlineAPI(agentId: number) {
  return http({ url: '/cs/agent/online', method: 'post', params: { agentId } })
}

export function agentOfflineAPI(agentId: number) {
  return http({ url: '/cs/agent/offline', method: 'post', params: { agentId } })
}

export function agentBusyAPI(agentId: number) {
  return http({ url: '/cs/agent/busy', method: 'post', params: { agentId } })
}

export function getOnlineAgentsAPI() {
  return http<Agent[]>({ url: '/cs/agent/online', method: 'get' })
}

export function getAgentInfoAPI(agentId: number) {
  return http<Agent>({ url: '/cs/agent/get', method: 'get', params: { agentId } })
}

export function getQuickReplyListAPI(groupId?: number) {
  return http<QuickReply[]>({ url: '/cs/quick-reply/list', method: 'get', params: groupId ? { groupId } : {} })
}

export function createQuickReplyAPI(data: QuickReply) {
  return http<QuickReply>({ url: '/cs/quick-reply/create', method: 'post', data })
}

export function deleteQuickReplyAPI(id: number) {
  return http({ url: '/cs/quick-reply/delete', method: 'delete', params: { id } })
}
