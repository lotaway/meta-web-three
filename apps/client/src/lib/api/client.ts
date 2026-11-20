import { ApiError, isApiError } from './errors'
import { getApiBaseUrl } from './baseUrl'

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

function buildRequestConfig(options: RequestInit): RequestInit {
  return {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  }
}

async function readErrorBody(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return response.text()
  }
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

  private throwTimeoutOrOriginal(error: unknown): never {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError('Request timed out', 'TIMEOUT_ERROR', 408)
    }
    throw error
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
      this.throwTimeoutOrOriginal(error)
    }
  }

  private async buildHandledError(
    response: Response,
    config: RequestInit
  ): Promise<ApiError> {
    const data = await readErrorBody(response)
    let handledError = ApiError.fromResponse(response, data)
    for (const interceptor of this.errorInterceptors) {
      handledError = await interceptor(handledError, config)
    }
    return handledError
  }

  private async parseSuccessBody(response: Response): Promise<unknown> {
    if (response.status === 204) {
      return null
    }
    return response.json()
  }

  private async executeRequestAttempt(
    url: string,
    config: RequestInit
  ): Promise<unknown> {
    let response = await this.fetchWithTimeout(
      `${this.options.baseUrl}${url}`,
      config,
      this.options.timeout
    )
    response = (await this.executeInterceptors(
      this.responseInterceptors as any,
      response as any
    )) as Response

    if (!response.ok) {
      throw await this.buildHandledError(response, config)
    }
    return this.parseSuccessBody(response)
  }

  private delayRetry(attempt: number): Promise<void> {
    return new Promise((resolve) =>
      setTimeout(resolve, this.options.retryDelay * attempt)
    )
  }

  private async handleRetryError(
    error: unknown,
    attempt: number,
    maxAttempts: number
  ): Promise<void> {
    if (attempt >= maxAttempts) {
      throw isApiError(error) ? error : ApiError.fromError(error)
    }
    if (isApiError(error) && error.status >= 400 && error.status < 500) {
      throw error
    }
    await this.delayRetry(attempt)
  }

  private async request<T = unknown>(
    url: string,
    options: RequestInit = {}
  ): Promise<T> {
    let config = buildRequestConfig(options)
    config = (await this.executeInterceptors(this.requestInterceptors, config)) as any

    let attempt = 0
    const maxAttempts = this.options.retryTimes + 1
    while (attempt < maxAttempts) {
      try {
        return (await this.executeRequestAttempt(url, config)) as T
      } catch (error) {
        attempt++
        await this.handleRetryError(error, attempt, maxAttempts)
      }
    }

    throw new ApiError('Request failed', 'UNKNOWN_ERROR', 0)
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
  baseUrl: getApiBaseUrl(),
  timeout: 10000,
  retryTimes: 2,
  retryDelay: 1000,
})