import { ApiClient, apiClient } from '../../../../src/lib/api/client'
import { ApiError } from '../../../../src/lib/api/errors'

const mockFetch = jest.fn()

jest.spyOn(global, 'fetch').mockImplementation(mockFetch)

describe('api/client', () => {
  let client: ApiClient

  beforeEach(() => {
    client = new ApiClient({
      baseUrl: 'http://localhost:3000',
      timeout: 5000,
      retryTimes: 2,
      retryDelay: 100,
    })
    jest.clearAllMocks()
  })

  describe('constructor', () => {
    it('should create client with default options', () => {
      const defaultClient = new ApiClient({ baseUrl: 'http://test.com' })
      expect(defaultClient).toBeDefined()
    })

    it('should create client with custom options', () => {
      const customClient = new ApiClient({
        baseUrl: 'http://test.com',
        timeout: 10000,
        retryTimes: 3,
        retryDelay: 500,
      })
      expect(customClient).toBeDefined()
    })
  })

  describe('interceptors', () => {
    it('should add request interceptor', () => {
      const interceptor = jest.fn((config) => config)
      client.addRequestInterceptor(interceptor)
      expect(client).toBeDefined()
    })

    it('should add response interceptor', () => {
      const interceptor = jest.fn((response) => response)
      client.addResponseInterceptor(interceptor)
      expect(client).toBeDefined()
    })

    it('should add error interceptor', () => {
      const interceptor = jest.fn((error) => error)
      client.addErrorInterceptor(interceptor)
      expect(client).toBeDefined()
    })
  })

  describe('request methods', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ code: '0000', data: 'test' }),
        headers: new Headers({ 'content-type': 'application/json' }),
      })
    })

    it('should make GET request', async () => {
      await client.get('/test')
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:3000/test',
        expect.objectContaining({ method: 'GET' })
      )
    })

    it('should make POST request', async () => {
      const data = { name: 'test' }
      await client.post('/test', data)
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:3000/test',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(data),
        })
      )
    })

    it('should make PUT request', async () => {
      const data = { name: 'updated' }
      await client.put('/test/1', data)
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:3000/test/1',
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify(data),
        })
      )
    })

    it('should make DELETE request', async () => {
      await client.delete('/test/1')
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:3000/test/1',
        expect.objectContaining({ method: 'DELETE' })
      )
    })

    it('should make PATCH request', async () => {
      const data = { partial: 'update' }
      await client.patch('/test/1', data)
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:3000/test/1',
        expect.objectContaining({
          method: 'PATCH',
          body: JSON.stringify(data),
        })
      )
    })
  })

  describe('error handling', () => {
    it('should throw ApiError for 400 response', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ code: '1000', message: '参数错误' }),
        headers: new Headers({ 'content-type': 'application/json' }),
      })

      await expect(client.get('/test')).rejects.toThrow(ApiError)
    })

    it('should throw ApiError for 404 response', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ code: '1005', message: '资源不存在' }),
        headers: new Headers({ 'content-type': 'application/json' }),
      })

      await expect(client.get('/test')).rejects.toThrow(ApiError)
    })

    it('should throw ApiError for 500 response', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ code: '9999', message: '系统错误' }),
        headers: new Headers({ 'content-type': 'application/json' }),
      })

      await expect(client.get('/test')).rejects.toThrow(ApiError)
    })

    it('should throw ApiError for network error', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'))

      await expect(client.get('/test')).rejects.toThrow(ApiError)
    })
  })

  describe('retry logic', () => {
    it('should retry on network error', async () => {
      mockFetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ code: '0000', data: 'success' }),
          headers: new Headers({ 'content-type': 'application/json' }),
        })

      const result = await client.get('/test')
      expect(result).toEqual({ code: '0000', data: 'success' })
    })

    it('should not retry on 4xx errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ code: '1000', message: 'Bad request' }),
        headers: new Headers({ 'content-type': 'application/json' }),
      })

      await expect(client.get('/test')).rejects.toThrow()
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })
  })

  describe('timeout', () => {
    it('should throw timeout error when request takes too long', async () => {
      // Use a real short timeout instead of fake timers: the mock must honour
      // the AbortSignal so fetchWithTimeout can convert the abort to ApiError.
      const shortTimeoutClient = new ApiClient({
        baseUrl: 'http://localhost:3000',
        timeout: 10,
        retryTimes: 0,
        retryDelay: 0,
      })
      mockFetch.mockImplementation(
        (_url: string, options?: RequestInit) =>
          new Promise((_resolve, reject) => {
            options?.signal?.addEventListener('abort', () => {
              const abortError = new Error('Aborted')
              abortError.name = 'AbortError'
              reject(abortError)
            })
          })
      )

      await expect(shortTimeoutClient.get('/test', {})).rejects.toThrow('请求超时')
    })
  })

  describe('apiClient singleton', () => {
    it('should have default base URL from env', () => {
      expect(apiClient).toBeDefined()
    })
  })
})