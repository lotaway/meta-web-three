import type { CommonPage } from '@/types/common'
import http from '@/utils/http'

// ==================== Admin API for Payment Management ====================

export interface UserKYC {
  id: number
  userId: number
  level: string
  status: string
  realName: string
  idNumber: string
  idType: string
  phoneNumber: string
  email: string
  address: string
  country: string
  nationality: string
  dateOfBirth: string
  gender: string
  idCardFrontUrl: string
  idCardBackUrl: string
  selfieUrl: string
  proofOfAddressUrl: string
  bankAccountNumber: string
  bankName: string
  bankBranch: string
  taxId: string
  occupation: string
  employer: string
  annualIncome: string
  sourceOfFunds: string
  purposeOfTransaction: string
  reviewerId: string
  reviewNotes: string
  submittedAt: string
  reviewedAt: string
  createdAt: string
  updatedAt: string
}

export interface CryptoPrice {
  id: number
  symbol: string
  baseCurrency: string
  quoteCurrency: string
  price: number
  bidPrice: number
  askPrice: number
  volume24h: number
  change24h: number
  changePercent24h: number
  source: string
  timestamp: string
}

export interface ExchangeOrder {
  id: number
  orderNo: string
  userId: number
  orderType: string
  status: string
  fiatCurrency: string
  cryptoCurrency: string
  fiatAmount: number
  cryptoAmount: number
  cryptoDecimals: number
  fee: number
  exchangeRate: number
  settlementAmount: number
  actualRate: number
  paymentMethod: string
  paymentOrderNo: string
  cryptoTransactionHash: string
  userWalletAddress: string
  failureReason: string
  paidAt: string
  completedAt: string
  expiredAt: string
  kycLevel: string
  kycVerified: boolean
  remark: string
  createdAt: string
  updatedAt: string
}

export interface UserKYCQueryParams {
  pageNum?: number
  pageSize?: number
  userId?: number
  level?: string
  status?: string
}

export interface CryptoPriceQueryParams {
  pageNum?: number
  pageSize?: number
  symbol?: string
  baseCurrency?: string
  quoteCurrency?: string
  source?: string
}

export interface ExchangeOrderQueryParams {
  pageNum?: number
  pageSize?: number
  userId?: number
  orderNo?: string
  status?: string
  orderType?: string
}

export interface KYCStatistics {
  total: number
  pending: number
  approved: number
  rejected: number
}

export interface ExchangeOrderStatistics {
  total: number
  pending: number
  completed: number
  failed: number
}

export interface PaymentStatistics {
  totalKYC: number
  pendingKYC: number
  totalOrders: number
  completedOrders: number
  cryptoSymbols: number
}

// Get KYC list (admin)
export function getKYCListAPI(params: UserKYCQueryParams) {
  return http<CommonPage<UserKYC>>({
    url: '/api/admin/payment/kyc/list',
    method: 'get',
    params,
  })
}

// Get KYC by ID (admin)
export function getKYCByIdAPI(id: number) {
  return http<UserKYC>({
    url: `/api/admin/payment/kyc/${id}`,
    method: 'get',
  })
}

// Review KYC (admin)
export function reviewKYCAPI(id: number, data: { status: string; reviewerId: string; reviewNotes: string }) {
  return http({
    url: `/api/admin/payment/kyc/${id}/review`,
    method: 'put',
    data,
  })
}

// Get KYC statistics (admin)
export function getKYCStatisticsAPI() {
  return http<KYCStatistics>({
    url: '/api/admin/payment/kyc/statistics',
    method: 'get',
  })
}

// Get crypto price list (admin)
export function getCryptoPriceListAPI(params: CryptoPriceQueryParams) {
  return http<CommonPage<CryptoPrice>>({
    url: '/api/admin/payment/crypto-price/list',
    method: 'get',
    params,
  })
}

// Get latest crypto prices (admin)
export function getLatestCryptoPricesAPI(baseCurrency?: string, quoteCurrency?: string) {
  return http<CryptoPrice[]>({
    url: '/api/admin/payment/crypto-price/latest',
    method: 'get',
    params: { baseCurrency, quoteCurrency },
  })
}

// Get exchange order list (admin)
export function getExchangeOrderListAPI(params: ExchangeOrderQueryParams) {
  return http<CommonPage<ExchangeOrder>>({
    url: '/api/admin/payment/exchange-order/list',
    method: 'get',
    params,
  })
}

// Get exchange order by ID (admin)
export function getExchangeOrderByIdAPI(id: number) {
  return http<ExchangeOrder>({
    url: `/api/admin/payment/exchange-order/${id}`,
    method: 'get',
  })
}

// Get exchange order statistics (admin)
export function getExchangeOrderStatisticsAPI() {
  return http<ExchangeOrderStatistics>({
    url: '/api/admin/payment/exchange-order/statistics',
    method: 'get',
  })
}

// Get payment statistics (admin)
export function getPaymentStatisticsAPI() {
  return http<PaymentStatistics>({
    url: '/api/admin/payment/statistics',
    method: 'get',
  })
}