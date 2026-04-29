import { Configuration } from '@/src/generated/api/runtime'
import { ProductManagementApi } from '@/src/generated/api/apis/ProductManagementApi'
import { ProductCategoryManagementApi } from '@/src/generated/api/apis/ProductCategoryManagementApi'
import { ProductBrandManagementApi } from '@/src/generated/api/apis/ProductBrandManagementApi'
import { OrderControllerApi } from '@/src/generated/api/apis/OrderControllerApi'
import { CartManagementApi } from '@/src/generated/api/apis/CartManagementApi'
import { PasskeyApi } from '@/src/generated/api/apis/PasskeyApi'
import { PayControllerApi } from '@/src/generated/api/apis/PayControllerApi'
import { UserControllerApi } from '@/src/generated/api/apis/UserControllerApi'
import { MemberAddressManagementApi } from '@/src/generated/api/apis/MemberAddressManagementApi'
import { AdvertiseManagementApi } from '@/src/generated/api/apis/AdvertiseManagementApi'

export const API_BASE_URL = process.env.NEXT_PUBLIC_BACK_API_HOST ?? 'http://localhost:10081'
export const DEFAULT_USER_ID = Number(process.env.EXPO_PUBLIC_DEFAULT_USER_ID ?? 1)
export const STRIPE_PUBLISHABLE_KEY = process.env.EXPO_PUBLIC_STRIPE_PUBLISHABLE_KEY ?? ''
export const WECHAT_APP_ID = process.env.EXPO_PUBLIC_WECHAT_APP_ID ?? ''
export const ALIPAY_APP_ID = process.env.EXPO_PUBLIC_ALIPAY_APP_ID ?? ''

export const apiConfig = new Configuration({
  basePath: API_BASE_URL,
  credentials: 'include',
})

export const productApi = new ProductManagementApi(apiConfig)
export const categoryApi = new ProductCategoryManagementApi(apiConfig)
export const brandApi = new ProductBrandManagementApi(apiConfig)
export const orderApi = new OrderControllerApi(apiConfig)
export const cartApi = new CartManagementApi(apiConfig)
export const passkeyApi = new PasskeyApi(apiConfig)
export const payApi = new PayControllerApi(apiConfig)
export const userApi = new UserControllerApi(apiConfig)
export const addressApi = new MemberAddressManagementApi(apiConfig)
export const advertiseApi = new AdvertiseManagementApi(apiConfig)

/**
 * Relying Party ID — 应与后端配置及 iOS associated domains 保持一致。
 * 开发期间可使用 localhost；生产环境改为实际域名（如 "example.com"）。
 */
export const RP_ID: string = process.env.EXPO_PUBLIC_RP_ID ?? 'localhost'
