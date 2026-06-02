import http from '@/utils/http'
import type { CommonResult, CommonPage } from '@/types/common'

export interface CommissionAccount {
  userId: number
  totalAmount: number
  availableAmount: number
  frozenAmount: number
}

export interface CommissionRecord {
  id: number
  userId: number
  orderId: number
  amount: number
  commissionLevel: number
  parentUserId: number
  status: string
  type: string
  createTime: string
  settleTime: string | null
}

export interface CommissionBindRequest {
  userId: number
  parentUserId: number
}

export interface CommissionCalcRequest {
  orderId: number
  userId: number
  payAmount: number
  availableAt: string
}

export interface CommissionSettleRequest {
  executeBefore: string
}

export interface CommissionCancelRequest {
  orderId: number
}

export interface CommissionBalanceQuery {
  userId: number
}

export interface CommissionLedgerQuery {
  userId: number
  page?: number
  size?: number
  status?: string
}

export function getCommissionBalanceAPI(userId: number) {
  return http<CommonResult<CommissionAccount>>({ url: '/commission/balance', method: 'get', params: { userId } })
}

export function getCommissionLedgerAPI(params: CommissionLedgerQuery) {
  return http<CommonResult<CommissionRecord[]>>({ url: '/commission/ledger', method: 'get', params })
}

export function bindCommissionRelationAPI(data: CommissionBindRequest) {
  return http<CommonResult<void>>({ url: '/commission/relations/bind', method: 'post', data })
}

export function calculateCommissionAPI(data: CommissionCalcRequest) {
  return http<CommonResult<void>>({ url: '/commission/calc', method: 'post', data })
}

export function settleCommissionAPI(data: CommissionSettleRequest) {
  return http<CommonResult<void>>({ url: '/commission/settle', method: 'post', data })
}

export function cancelCommissionAPI(data: CommissionCancelRequest) {
  return http<CommonResult<void>>({ url: '/commission/cancel', method: 'post', data })
}