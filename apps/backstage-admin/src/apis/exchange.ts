import http from '@/utils/http'

export interface ExchangeRate {
  id?: number
  sourceCurrency: string
  targetCurrency: string
  rate: number
  effectiveDate: string
  rateType: 'SPOT' | 'MIDDLE' | 'SELLING' | 'BUYING'
  isActive: boolean
  createdBy?: string
  createdAt?: string
  updatedAt?: string
}

export const getActiveRates = () => {
  return http<ExchangeRate[]>({ url: '/api/exchange-rates', method: 'get' })
}

export const getExchangeRateById = (id: number) => {
  return http<ExchangeRate>({ url: `/api/exchange-rates/${id}`, method: 'get' })
}

export const getRatesByCurrencyPair = (sourceCurrency: string, targetCurrency: string) => {
  return http<ExchangeRate[]>({
    url: '/api/exchange-rates/pair',
    method: 'get',
    params: { sourceCurrency, targetCurrency }
  })
}

export const getEffectiveRate = (
  sourceCurrency: string,
  targetCurrency: string,
  date: string
) => {
  return http<ExchangeRate>({
    url: '/api/exchange-rates/effective',
    method: 'get',
    params: { sourceCurrency, targetCurrency, date }
  })
}

export interface CreateExchangeRateRequest {
  sourceCurrency: string
  targetCurrency: string
  rate: number
  effectiveDate: string
  rateType: string
  createdBy: string
}

export const createExchangeRate = (data: CreateExchangeRateRequest) => {
  return http<ExchangeRate>({ url: '/api/exchange-rates', method: 'post', data })
}

export const updateExchangeRate = (id: number, rate: number) => {
  return http<ExchangeRate>({
    url: `/api/exchange-rates/${id}`,
    method: 'put',
    data: { rate }
  })
}

export const deleteExchangeRate = (id: number) => {
  return http({ url: `/api/exchange-rates/${id}`, method: 'delete' })
}