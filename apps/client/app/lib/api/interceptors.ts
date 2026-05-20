import { apiClient } from './client'
import { ApiError, getErrorMessage } from './errors'

const TOKEN_KEY = 'auth_token'

export function getToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(TOKEN_KEY, token)
}

export function removeToken(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(TOKEN_KEY)
}

apiClient.addRequestInterceptor((config) => {
  const token = getToken()
  if (token) {
    config.headers = {
      ...config.headers,
      Authorization: `Bearer ${token}`,
    }
  }

  console.log(`[API Request] ${config.method || 'GET'}`, config.url)

  return config
})

apiClient.addResponseInterceptor(async (response, config) => {
  console.log(`[API Response] ${response.status} ${config.url}`)

  if (response.headers.get('content-type')?.includes('application/json')) {
    const data = await response.clone().json()

    if (data.code && data.code !== '0000') {
      console.warn(`[API Warning] ${data.code}: ${data.message}`)
    }
  }

  return response
})

apiClient.addErrorInterceptor(async (error, config) => {
  console.error(`[API Error] ${error.code}: ${error.message}`)

  if (error.code === '2003' || error.code === '2004') {
    removeToken()
    console.warn('Token expired, please login again')

    if (typeof window !== 'undefined') {
      window.location.href = '/login'
    }
  }

  if (error.status === 401) {
    removeToken()

    if (typeof window !== 'undefined') {
      window.location.href = '/login'
    }
  }

  if (error.status === 403) {
    console.warn('Access denied')
  }

  if (error.status === 404) {
    console.warn('Resource not found')
  }

  if (error.status && error.status >= 500) {
    console.error('Server error, please try again later')
  }

  return error
})

export function initializeApiInterceptors() {
  console.log('[API] Interceptors initialized')
}