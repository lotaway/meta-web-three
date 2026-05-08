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
import { ESProductManagementApi } from "@/src/generated/api/apis/ESProductManagementApi"
import { CouponControllerApi } from '@/src/generated/api/apis/CouponControllerApi'
import { UserActionControllerApi } from '@/src/generated/api/apis/UserActionControllerApi'
import { NotificationControllerApi } from '@/src/generated/api/apis/NotificationControllerApi'
import { SsoControllerApi } from '@/src/generated/api/apis/SsoControllerApi'

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
export const esProductApi = new ESProductManagementApi(apiConfig)
export const couponApi = new CouponControllerApi(apiConfig)
export const productCollectionApi = new UserActionControllerApi(apiConfig)
export const commentApi = new UserActionControllerApi(apiConfig)
export const notificationApi = new NotificationControllerApi(apiConfig)
export const ssoApi = new SsoControllerApi(apiConfig)

export const stripeKey = STRIPE_PUBLISHABLE_KEY
