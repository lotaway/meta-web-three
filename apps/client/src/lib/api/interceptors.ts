import { apiClient } from './client'
import { clearToken, getToken } from '@/lib/auth/tokenStorage'

export { getToken, setToken, clearToken } from '@/lib/auth/tokenStorage'

apiClient.addRequestInterceptor(async (config: any) => {
  const token = await getToken()
  if (token) {
    config.headers = {
      ...config.headers,
      Authorization: `Bearer ${token}`,
    }
  }

  console.log(`[API Request] ${config.method || 'GET'}`, config.url)

  return config
})

apiClient.addResponseInterceptor(async (response: any, config: any) => {
  console.log(`[API Response] ${response.status} ${config?.url}`)

  if (response.headers.get('content-type')?.includes('application/json')) {
    const data = await response.clone().json()

    if (data.code && data.code !== '0000') {
      console.warn(`[API Warning] ${data.code}: ${data.message}`)
    }
  }

  return response
})

const redirectToLogin = () => {
  if (typeof window !== 'undefined') window.location.href = '/login'
}

const handleTokenExpiration = async (error: any) => {
  if (error.code !== '2003' && error.code !== '2004') return
  await clearToken()
  console.warn('Token expired, please login again')
  redirectToLogin()
}

const handleUnauthorized = async () => {
  await clearToken()
  redirectToLogin()
}

const handleForbidden = () => {
  console.warn('Access denied')
}

const handleNotFound = () => {
  console.warn('Resource not found')
}

const handleServerError = () => {
  console.error('Server error, please try again later')
}

const statusHandlers: Record<string, () => void | Promise<void>> = {
  401: handleUnauthorized,
  403: handleForbidden,
  404: handleNotFound,
}

const handleStatus = (error: any) => {
  const status = Number(error.status)
  const handler = Number.isFinite(status) ? statusHandlers[status] : undefined
  if (handler) return handler()
  if (status >= 500) handleServerError()
}

apiClient.addErrorInterceptor(async (error: any, config: any) => {
  console.error(`[API Error] ${error.code}: ${error.message}`)
  await handleTokenExpiration(error)
  await handleStatus(error)
  return error
})

export function initializeApiInterceptors() {
  console.log('[API] Interceptors initialized')
}
