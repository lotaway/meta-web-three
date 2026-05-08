import { useUserStore } from '@/stores/user'
import type { CommonResult } from '@/types/common'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'

// 服务前缀映射表 — 根据 URL 首段匹配微服务
const SERVICE_PREFIX_MAP: Record<string, string> = {
  '/product': 'product-service',
  '/brand': 'product-service',
  '/productCategory': 'product-service',
  '/productAttribute': 'product-service',
  '/sku': 'product-service',
  '/subject': 'product-service',
  '/prefrenceArea': 'product-service',
  '/order': 'order-service',
  '/returnReason': 'order-service',
  '/returnApply': 'order-service',
  '/companyAddress': 'order-service',
  '/orderSetting': 'order-service',
  '/coupon': 'promotion-service',
  '/flash': 'promotion-service',
  '/home': 'promotion-service',
  '/admin': 'user-service',
  '/role': 'user-service',
  '/menu': 'user-service',
  '/resource': 'user-service',
  '/memberLevel': 'user-service',
  '/sso': 'user-service',
  '/member': 'user-service',
}

// 创建axios实例
const http = axios.create({
  baseURL: import.meta.env.VITE_BASE_SERVER_URL,
  timeout: 5000,
})

// 请求拦截器 1：自动添加微服务前缀（最先生效）
http.interceptors.request.use(
  config => {
    const url = config.url || ''
    const matchedPrefix = Object.keys(SERVICE_PREFIX_MAP)
      .sort((a, b) => b.length - a.length) // 长路径优先，避免 /home 误匹配 /home/advertise 等
      .find(prefix => url.startsWith(prefix) || url.startsWith('/' + prefix))
    if (matchedPrefix) {
      const serviceId = SERVICE_PREFIX_MAP[matchedPrefix]
      config.url = `/${serviceId}${url.startsWith('/') ? '' : '/'}${url}`
    }
    return config
  },
  e => Promise.reject(e),
)

// axios请求拦截器 2：添加 token 鉴权头
http.interceptors.request.use(
  config => {
    //从pinia获取token
    const userStore = useUserStore()
    const token = userStore.userInfo.token
    if (token) {
      config.headers.Authorization = token
    }
    return config
  },
  e => Promise.reject(e),
)

// axios响应拦截器
http.interceptors.response.use(
  response => {
    const res: CommonResult<unknown> = response.data
    // 后端 code 为字符串(如 "200")，前端接受 number 和 string
    const code = String(res.code)
    if (code !== '200') {
      ElMessage({
        message: res.message,
        type: 'error',
        duration: 3 * 1000,
      })
      if (code === '401') {
        ElMessageBox.confirm('你已被登出，可以取消继续留在该页面，或者重新登录', '确定登出', {
          confirmButtonText: '重新登录',
          cancelButtonText: '取消',
          type: 'warning',
        }).then(() => {
          const userStore = useUserStore()
          userStore.fedLogout()
          // 为了重新实例化vue-router对象 避免bug
          location.reload()
        })
      }
      return Promise.reject('error')
    } else {
      // 返回响应JSON中的data属性，不包括message和code
      return response.data
    }
  },
  error => {
    // 全局处理异常请求
    console.log('error' + error)
    ElMessage({
      message: error.message,
      type: 'error',
      duration: 3 * 1000,
    })
    return Promise.reject(error)
  },
)

export default http
