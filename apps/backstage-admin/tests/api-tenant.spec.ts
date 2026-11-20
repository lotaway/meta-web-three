import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { AxiosRequestConfig } from 'axios'

const h = vi.hoisted(() => ({
  http: vi.fn(),
}))

vi.mock('@/utils/http', () => ({ default: h.http }))

import {
  approveTenantAPI,
  disableTenantAPI,
  getShopByTenantAPI,
  getTenantByIdAPI,
  getTenantListAPI,
  rejectTenantAPI,
  updateShopAPI,
} from '@/apis/tenant'

const RESULT = { code: '0000', data: { id: 1 }, message: '' }

function lastCall(): AxiosRequestConfig {
  return h.http.mock.calls[h.http.mock.calls.length - 1][0]
}

describe('tenant API request construction', () => {
  beforeEach(() => {
    h.http.mockReset()
    h.http.mockResolvedValue(RESULT)
  })

  it('maps pageNum/pageSize to gateway page/size with defaults', async () => {
    await expect(getTenantListAPI({})).resolves.toBe(RESULT)
    expect(lastCall()).toMatchObject({
      url: '/tenant',
      method: 'get',
      params: { page: 1, size: 10, status: undefined },
    })
  })

  it('passes explicit pagination and status through', async () => {
    await getTenantListAPI({ pageNum: 3, pageSize: 25, status: 'ACTIVE' })
    expect(lastCall().params).toEqual({ page: 3, size: 25, status: 'ACTIVE' })
  })

  it('builds id-based urls for read and state transitions', async () => {
    await getTenantByIdAPI(42)
    expect(lastCall()).toMatchObject({ url: '/tenant/42', method: 'get' })

    await approveTenantAPI(42)
    expect(lastCall()).toMatchObject({ url: '/tenant/42/approve', method: 'post' })

    await rejectTenantAPI(42)
    expect(lastCall()).toMatchObject({ url: '/tenant/42/reject', method: 'post' })

    await disableTenantAPI(42)
    expect(lastCall()).toMatchObject({ url: '/tenant/42/disable', method: 'post' })
  })

  it('reads and updates the tenant shop with the correct methods', async () => {
    const shop = { name: 'flagship', status: 'OPEN' }
    await getShopByTenantAPI(7)
    expect(lastCall()).toMatchObject({ url: '/tenant/7/shop', method: 'get' })

    await updateShopAPI(7, shop)
    expect(lastCall()).toMatchObject({
      url: '/tenant/7/shop',
      method: 'put',
      data: shop,
    })
  })

  it('returns the http result unchanged to callers', async () => {
    await expect(getTenantByIdAPI(1)).resolves.toBe(RESULT)
  })
})
