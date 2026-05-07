import { ApiError, isApiError } from './errors'

type RequestInterceptor = (config: RequestInit) => RequestInit | Promise<RequestInit>
type ResponseInterceptor = (
  response: Response,
  config: RequestInit
) => Response | Promise<Response>
type ErrorInterceptor = (
  error: ApiError,
  config: RequestInit
) => ApiError | Promise<ApiError>

export interface ApiClientOptions {
  baseUrl: string
  timeout?: number
  onRequest?: RequestInterceptor
  onResponse?: ResponseInterceptor
  onError?: ErrorInterceptor
  retryTimes?: number
  retryDelay?: number
}

const defaultOptions: Required<Omit<ApiClientOptions, 'baseUrl'>> = {
  timeout: 10000,
  onRequest: (config) => config,
  onResponse: (response) => response,
  onError: (error) => error,
  retryTimes: 0,
  retryDelay: 1000,
}

export class ApiClient {
  private options: Required<ApiClientOptions>
  private requestInterceptors: RequestInterceptor[] = []
  private responseInterceptors: ResponseInterceptor[] = []
  private errorInterceptors: ErrorInterceptor[] = []

  constructor(options: ApiClientOptions) {
    this.options = {
      ...defaultOptions,
      ...options,
    }
  }

  addRequestInterceptor(interceptor: RequestInterceptor): void {
    this.requestInterceptors.push(interceptor)
  }

  addResponseInterceptor(interceptor: ResponseInterceptor): void {
    this.responseInterceptors.push(interceptor)
  }

  addErrorInterceptor(interceptor: ErrorInterceptor): void {
    this.errorInterceptors.push(interceptor)
  }

  private async executeInterceptors<T>(
    interceptors: ((config: T) => T | Promise<T>)[],
    value: T
  ): Promise<T> {
    let result = value
    for (const interceptor of interceptors) {
      result = await interceptor(result)
    }
    return result
  }

  private async fetchWithTimeout(
    url: string,
    options: RequestInit,
    timeout: number
  ): Promise<Response> {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      })
      clearTimeout(timeoutId)
      return response
    } catch (error) {
      clearTimeout(timeoutId)
      if (error instanceof Error && error.name === 'AbortError') {
        throw new ApiError('请求超时', 'TIMEOUT_ERROR', 408)
      }
      throw error
    }
  }

  private async request<T = unknown>(
    url: string,
    options: RequestInit = {}
  ): Promise<T> {
    let config = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    }

    config = await this.executeInterceptors(this.requestInterceptors, config)

    let response: Response
    let attempt = 0
    const maxAttempts = this.options.retryTimes + 1

    while (attempt < maxAttempts) {
      try {
        response = await this.fetchWithTimeout(
          `${this.options.baseUrl}${url}`,
          config,
          this.options.timeout
        )

        response = await this.executeInterceptors(
          this.responseInterceptors,
          response
        )

        if (!response.ok) {
          let data: unknown = null
          try {
            data = await response.json()
          } catch {
            data = await response.text()
          }

          const apiError = ApiError.fromResponse(response, data)
          let handledError = apiError

          for (const interceptor of this.errorInterceptors) {
            handledError = await interceptor(handledError, config)
          }

          throw handledError
        }

        if (response.status === 204) {
          return null as T
        }

        const data = await response.json()
        return data as T
      } catch (error) {
        attempt++

        if (attempt >= maxAttempts) {
          throw error
        }

        if (isApiError(error) && error.status >= 400 && error.status < 500) {
          throw error
        }

        await new Promise((resolve) =>
          setTimeout(resolve, this.options.retryDelay * attempt)
        )
      }
    }

    throw new ApiError('请求失败', 'UNKNOWN_ERROR', 0)
  }

  get<T = unknown>(url: string, options?: RequestInit): Promise<T> {
    return this.request<T>(url, { ...options, method: 'GET' })
  }

  post<T = unknown>(url: string, body?: unknown, options?: RequestInit): Promise<T> {
    return this.request<T>(url, {
      ...options,
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
    })
  }

  put<T = unknown>(url: string, body?: unknown, options?: RequestInit): Promise<T> {
    return this.request<T>(url, {
      ...options,
      method: 'PUT',
      body: body ? JSON.stringify(body) : undefined,
    })
  }

  delete<T = unknown>(url: string, options?: RequestInit): Promise<T> {
    return this.request<T>(url, { ...options, method: 'DELETE' })
  }

  patch<T = unknown>(url: string, body?: unknown, options?: RequestInit): Promise<T> {
    return this.request<T>(url, {
      ...options,
      method: 'PATCH',
      body: body ? JSON.stringify(body) : undefined,
    })
  }
}

export const apiClient = new ApiClient({
  baseUrl: process.env.NEXT_PUBLIC_BACK_API_HOST ?? 'http://localhost:10081',
  timeout: 10000,
  retryTimes: 2,
  retryDelay: 1000,
})