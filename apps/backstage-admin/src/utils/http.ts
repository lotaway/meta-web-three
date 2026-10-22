import { useUserStore } from '@/stores/user'
import type { CommonResult } from '@/types/common'
import axios, { type AxiosInstance, type AxiosResponse, type InternalAxiosRequestConfig, type AxiosRequestConfig } from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'
import { HTTP_TIMEOUT, MESSAGE_DURATION } from '@/constants'
import { BASE_SERVER_URL } from '@/utils/env'
import { t } from '@/locales'

/**
 * 前端路径前缀 -> 后端服务 ID 映射。
 *
 * <p>网关开启 Spring Cloud Gateway discovery locator 后，请求路径需带
 * `/{serviceId}` 前缀（如 `/mall-catalog-service/product/list`），
 * 网关据此把流量路由到对应微服务。缺少前缀会导致 404。</p>
 */
const SERVICE_PREFIX_MAP: Record<string, string> = {
  '/product': 'mall-catalog-service',
  '/brand': 'mall-catalog-service',
  '/product-category': 'mall-catalog-service',
  '/productAttribute': 'mall-catalog-service',
  '/sku': 'mall-catalog-service',
  '/subject': 'mall-catalog-service',
  '/prefrenceArea': 'mall-trade-service',
  '/order': 'mall-trade-service',
  '/returnReason': 'mall-trade-service',
  '/returnApply': 'mall-trade-service',
  '/companyAddress': 'mall-trade-service',
  '/orderSetting': 'mall-trade-service',
  '/coupon': 'mall-trade-service',
  '/flash': 'mall-trade-service',
  '/home': 'mall-trade-service',
  '/admin': 'mall-user-service',
  '/role': 'mall-user-service',
  '/menu': 'mall-user-service',
  '/resource': 'mall-user-service',
  '/memberLevel': 'mall-user-service',
  '/sso': 'mall-user-service',
  '/member': 'mall-user-service',
  '/tenant': 'platform-service',
  '/developer': 'platform-service',
  '/cs': 'platform-service',
  '/api/mes': 'factory-service',
  '/api/pokayoke': 'factory-service',
  '/api/admin/ai-shopping': 'mall-catalog-service',
  '/api/forecasting': 'ai-service',
  '/api/v1/route-optimizer': 'ai-service',
  // The merged project controllers kept their original base path (/project-service/*)
  '/project-service': 'erp-management-service',
  '/api/digital-twin': 'factory-service',
  '/api/alert-rules': 'factory-service',
  '/api/v1/production': 'factory-service',
  '/api/crm': 'erp-management-service',
  '/api/hrm': 'erp-management-service',
  '/api/finance': 'finance-service',
  '/api/settlement': 'finance-service',
  '/api/budget': 'finance-service',
  '/api/social-commerce': 'mall-social-service',
  '/api/group-buying': 'mall-social-service',
  '/api/review': 'mall-catalog-service',
  '/api/admin/risk': 'mall-user-service',
  '/api/admin/user': 'mall-user-service',
  '/api/admin/user-action': 'mall-user-service',
  '/api/traceability': 'blockchain-service',
  '/api/v1/wallets': 'blockchain-service',
  '/commission': 'platform-service',
}

const service: AxiosInstance = axios.create({
  baseURL: BASE_SERVER_URL,
  timeout: HTTP_TIMEOUT,
})

service.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const url = config.url || ''
    const matchedPrefix = Object.keys(SERVICE_PREFIX_MAP)
      .sort((a, b) => b.length - a.length)
      .find(prefix => url.startsWith(prefix) || url.startsWith('/' + prefix))
    if (matchedPrefix) {
      const serviceId = SERVICE_PREFIX_MAP[matchedPrefix]
      config.url = `/${serviceId}${url.startsWith('/') ? '' : '/'}${url}`
    }
    const userStore = useUserStore()
    const token = userStore.userInfo.token
    if (token) {
      config.headers.Authorization = token
    }
    return config
  },
  e => Promise.reject(e),
)

// 响应拦截器
service.interceptors.response.use(
  (response: AxiosResponse) => {
    const res = response.data as CommonResult<unknown>
    const code = String(res.code)
    if (code !== '0000') {
      ElMessage({
        message: res.message,
        type: 'error',
        duration: MESSAGE_DURATION,
      })
      if (code === '401') {
        ElMessageBox.confirm(t('http.logoutMessage'), t('http.logoutTitle'), {
          confirmButtonText: t('http.reLogin'),
          cancelButtonText: t('http.stayCurrentPage'),
          type: 'warning',
        }).then(() => {
          const userStore = useUserStore()
          userStore.fedLogout()
          location.reload()
        })
      }
      return Promise.reject(new Error(res.message))
    }
    // 返回完整响应对象，包含 .data 属性供调用者访问实际数据
    return response
  },
  error => {
    ElMessage({
      message: error.message || 'Request failed',
      type: 'error',
      duration: MESSAGE_DURATION,
    })
    return Promise.reject(error)
  },
)

// http 函数 - 支持泛型
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function http<T = any>(config: AxiosRequestConfig): Promise<CommonResult<T>> {
  // @ts-ignore - TypeScript 无法正确推断拦截器返回类型
  return service(config).then((response: AxiosResponse<CommonResult<T>>) => {
    return response.data as CommonResult<T>
  })
}

// 添加便捷方法
http.get = <T = any>(url: string, config?: AxiosRequestConfig): Promise<CommonResult<T>> => {
  return http<T>({ ...config, method: 'GET', url })
}

http.post = <T = any>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<CommonResult<T>> => {
  return http<T>({ ...config, method: 'POST', url, data })
}

http.put = <T = any>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<CommonResult<T>> => {
  return http<T>({ ...config, method: 'PUT', url, data })
}

http.delete = <T = any>(url: string, config?: AxiosRequestConfig): Promise<CommonResult<T>> => {
  return http<T>({ ...config, method: 'DELETE', url })
}

http.patch = <T = any>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<CommonResult<T>> => {
  return http<T>({ ...config, method: 'PATCH', url, data })
}

export default http