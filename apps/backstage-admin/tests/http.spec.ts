import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { AxiosRequestConfig } from 'axios'

// Capture the interceptors http.ts registers so we can exercise them
// deterministically without a live axios adapter.
const h = vi.hoisted(() => ({
  reqFulfilled: undefined as ((config: AxiosRequestConfig) => AxiosRequestConfig) | undefined,
  resFulfilled: undefined as ((response: { data: unknown }) => unknown) | undefined,
  resRejected: undefined as ((error: unknown) => unknown) | undefined,
  createConfig: undefined as AxiosRequestConfig | undefined,
  rawRequest: vi.fn(),
  userInfo: { token: '' } as { token: string },
  ElMessage: vi.fn(),
  confirm: vi.fn(),
}))

vi.mock('axios', () => ({
  default: {
    create: (config: AxiosRequestConfig) => {
      h.createConfig = config
      return {
        interceptors: {
          request: { use: (fn: typeof h.reqFulfilled) => (h.reqFulfilled = fn) },
          response: {
            use: (ok: typeof h.resFulfilled, err: typeof h.resRejected) => {
              h.resFulfilled = ok
              h.resRejected = err
            },
          },
        },
        // Emulate only the response interceptor chain here; the request
        // interceptor is asserted directly via applyRequest(). The two-arg
        // then() makes resRejected handle transport failures only, and its
        // returned promise is adopted (not re-wrapped) so rejections propagate.
        request: (cfg: AxiosRequestConfig) =>
          h.rawRequest(cfg).then(
            (response) => (h.resFulfilled ? h.resFulfilled(response as never) : response),
            (error) => (h.resRejected ? h.resRejected(error) : Promise.reject(error))
          ),
      }
    },
  },
}))

vi.mock('@/stores/user', () => ({
  useUserStore: () => ({
    userInfo: h.userInfo,
    fedLogout: vi.fn(),
  }),
}))

vi.mock('@/utils/env', () => ({ BASE_SERVER_URL: 'http://gateway.test:10081' }))
vi.mock('@/constants', () => ({ HTTP_TIMEOUT: 12345, MESSAGE_DURATION: 3000 }))
vi.mock('@/locales', () => ({ t: (key: string) => key }))
vi.mock('element-plus', () => ({
  ElMessage: h.ElMessage,
  ElMessageBox: { confirm: h.confirm },
}))

import http from '@/utils/http'

function applyRequest(config: AxiosRequestConfig): AxiosRequestConfig {
  return h.reqFulfilled!(config)
}

function respondWith(code: string, data: unknown = null, message = 'error'): Promise<unknown> {
  h.rawRequest.mockResolvedValueOnce({ data: { code, data, message } })
  return http({ url: '/tenant' })
}

describe('http client bootstrap', () => {
  it('uses the gateway base url and configured timeout', () => {
    expect(h.createConfig).toMatchObject({
      baseURL: 'http://gateway.test:10081',
      timeout: 12345,
    })
  })
})

describe('gateway service prefix mapping', () => {
  beforeEach(() => {
    h.userInfo.token = ''
  })

  it('prefixes a known frontend path with the owning service id', () => {
    expect(applyRequest({ url: '/product/list' }).url).toBe(
      '/mall-catalog-service/product/list'
    )
    expect(applyRequest({ url: '/tenant/1' }).url).toBe('/platform-service/tenant/1')
    expect(applyRequest({ url: '/api/mes/scada/telemetry' }).url).toBe(
      '/factory-service/api/mes/scada/telemetry'
    )
  })

  it('prefers the longest matching prefix', () => {
    // /api/admin/user and /api/admin/user-action both exist; the longer one
    // must win so the resulting url never carries a duplicated service id.
    const url = applyRequest({ url: '/api/admin/user-action/list' }).url
    expect(url).toBe('/mall-user-service/api/admin/user-action/list')
  })

  it('leaves an unmapped path untouched', () => {
    expect(applyRequest({ url: 'product/list' }).url).toBe('product/list')
    expect(applyRequest({ url: 'https://third.party/api' }).url).toBe('https://third.party/api')
  })

  it('injects the token header only when the user is logged in', () => {
    h.userInfo.token = 'Bearer abc'
    const authed = applyRequest({ url: '/tenant', headers: {} as never })
    expect(authed.headers?.Authorization).toBe('Bearer abc')

    h.userInfo.token = ''
    const anon = applyRequest({ url: '/tenant', headers: {} as never })
    expect(anon.headers?.Authorization).toBeUndefined()
  })
})

describe('http response handling', () => {
  beforeEach(() => {
    h.rawRequest.mockReset()
    h.ElMessage.mockClear()
    h.confirm.mockClear()
  })

  it('returns the unwrapped CommonResult on success code', async () => {
    const result = await respondWith('0000', { id: 7 })
    expect(result).toEqual({ code: '0000', data: { id: 7 }, message: 'error' })
  })

  it('rejects with an Error carrying the backend message on failure code', async () => {
    await expect(respondWith('1001', null, 'bad request')).rejects.toThrow('bad request')
    expect(h.ElMessage).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'bad request', type: 'error' })
    )
  })

  it('reports transport errors through ElMessage and rejects', async () => {
    h.rawRequest.mockRejectedValueOnce(new Error('Network down'))
    await expect(http({ url: '/tenant' })).rejects.toThrow('Network down')
    expect(h.ElMessage).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Network down', type: 'error' })
    )
  })
})

describe('http convenience methods', () => {
  beforeEach(() => {
    h.rawRequest.mockReset()
    h.rawRequest.mockResolvedValue({ data: { code: '0000', data: null } })
  })

  it.each([
    ['get', 'GET', undefined],
    ['post', 'POST', { a: 1 }],
    ['put', 'PUT', { a: 1 }],
    ['patch', 'PATCH', { a: 1 }],
  ] as const)('%s sends the expected method and body', async (method, httpMethod, body) => {
    await (http as any)[method]('/tenant/1', body)
    expect(h.rawRequest).toHaveBeenCalledWith(
      expect.objectContaining({ method: httpMethod, url: '/tenant/1' })
    )
    if (body) {
      expect(h.rawRequest).toHaveBeenCalledWith(expect.objectContaining({ data: body }))
    }
  })

  it('delete sends DELETE without a body', async () => {
    await http.delete('/tenant/1')
    expect(h.rawRequest).toHaveBeenCalledWith(
      expect.objectContaining({ method: 'DELETE', url: '/tenant/1' })
    )
  })
})
