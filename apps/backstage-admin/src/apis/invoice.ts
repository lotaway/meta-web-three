import type { CommonPage } from '@/types/common'
import http from '@/utils/http'

export interface InvoiceDTO {
  id?: number
  invoiceNo?: string
  orderNo?: string
  customerId?: number
  customerName?: string
  customerTaxNo?: string
  type?: string
  amount?: number
  taxRate?: string
  status?: string
  issuer?: string
  issueDate?: string
  printDate?: string
  voidDate?: string
  voidReason?: string
  redFlushDate?: string
  redFlushReason?: string
  createTime?: string
  createBy?: string
}

export interface InvoiceRequest {
  invoiceNo: string
  orderNo: string
  customerId: number
  customerName: string
  customerTaxNo: string
  type: string
  amount: number
  taxRate: string
}

// Invoice List APIs
export function getInvoiceListAPI(params: {
  pageNum?: number
  pageSize?: number
  status?: string
  customerId?: number
  startDate?: string
  endDate?: string
}) {
  return http<CommonPage<InvoiceDTO>>({
    url: '/api/invoice',
    method: 'get',
    params: params,
  })
}

export function getInvoiceByIdAPI(id: number) {
  return http<InvoiceDTO>({
    url: '/api/invoice/' + id,
    method: 'get',
  })
}

export function createInvoiceAPI(data: InvoiceRequest) {
  return http<number>({
    url: '/api/invoice',
    method: 'post',
    data: data,
  })
}

export function issueInvoiceAPI(id: number, issuer: string) {
  return http({
    url: '/api/invoice/' + id + '/issue',
    method: 'post',
    params: { issuer },
  })
}

export function printInvoiceAPI(id: number) {
  return http({
    url: '/api/invoice/' + id + '/print',
    method: 'post',
  })
}

export function voidInvoiceAPI(id: number, reason: string) {
  return http({
    url: '/api/invoice/' + id + '/void',
    method: 'post',
    params: { reason },
  })
}

export function redFlushInvoiceAPI(id: number, reason: string) {
  return http({
    url: '/api/invoice/' + id + '/red-flush',
    method: 'post',
    params: { reason },
  })
}