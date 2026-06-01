import http from '@/utils/http'

export interface Wallet {
  id: number
  userId: string
  chainType: 'ETHEREUM' | 'POLYGON' | 'BSC' | 'SOLANA' | 'POLKADOT'
  address: string
  balance: number
  status: 'ACTIVE' | 'FROZEN' | 'CLOSED'
  createdAt: string
  updatedAt?: string
}

export interface WalletTransaction {
  id: number
  walletId: number
  type: 'DEPOSIT' | 'WITHDRAW' | 'TRANSFER_IN' | 'TRANSFER_OUT'
  amount: number
  fee?: number
  txHash?: string
  status: 'PENDING' | 'CONFIRMED' | 'FAILED'
  createdAt: string
}

export interface WalletQueryParam {
  pageNum?: number
  pageSize?: number
  userId?: string
  chainType?: string
  status?: string
}

// Wallet APIs
export function createWalletAPI(data: { userId: string; chainType: string; address: string }) {
  return http<Wallet>({ url: '/api/v1/wallets', method: 'post', data })
}

export function getWalletByIdAPI(id: number) {
  return http<Wallet>({ url: `/api/v1/wallets/${id}`, method: 'get' })
}

export function getWalletByUserIdAPI(userId: string) {
  return http<Wallet[]>({ url: '/api/v1/wallets', method: 'get', params: { userId } })
}

export function getWalletsAPI(params?: WalletQueryParam) {
  return http<Wallet[]>({ url: '/api/v1/wallets', method: 'get', params })
}

export function depositAPI(id: number, amount: number) {
  return http<Wallet>({ url: `/api/v1/wallets/${id}/deposit`, method: 'post', data: { amount } })
}

export function withdrawAPI(id: number, amount: number) {
  return http<Wallet>({ url: `/api/v1/wallets/${id}/withdraw`, method: 'post', data: { amount } })
}

export function freezeWalletAPI(id: number) {
  return http<Wallet>({ url: `/api/v1/wallets/${id}/freeze`, method: 'post' })
}

export function activateWalletAPI(id: number) {
  return http<Wallet>({ url: `/api/v1/wallets/${id}/activate`, method: 'post' })
}

// Transaction APIs
export function getWalletTransactionsAPI(walletId: number, params?: { pageNum?: number; pageSize?: number }) {
  return http<WalletTransaction[]>({
    url: `/api/v1/wallets/${walletId}/transactions`,
    method: 'get',
    params,
  })
}

export function getTransactionByTxHashAPI(txHash: string) {
  return http<WalletTransaction>({ url: `/api/v1/transactions/${txHash}`, method: 'get' })
}