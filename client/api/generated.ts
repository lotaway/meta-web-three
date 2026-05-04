import { Configuration, BaseAPI, HTTPHeaders, QueryParameter } from '@/src/generated/api/runtime'
import * as runtime from '@/src/generated/api/runtime'
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
import { ApiResponseUserDTO, ApiResponseVoid } from '@/src/generated/api/models'

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

export interface SsoLoginRequest {
  username: string
  password: string
}

export interface SsoRegisterRequest {
  username: string
  password: string
  telephone?: string
  authCode?: string
}

export interface SsoInfoRequest {
  xUserId: number
}

export interface SsoLoginResponse {
  token: string
  tokenHead: string
}

export class SsoApi extends BaseAPI {
  async loginRaw(requestParameters: SsoLoginRequest, initOverrides?: RequestInit) {
    if (requestParameters['username'] == null) {
      throw new Error('Required parameter "username" was null or undefined')
    }
    if (requestParameters['password'] == null) {
      throw new Error('Required parameter "password" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    queryParameters['username'] = requestParameters['username']
    queryParameters['password'] = requestParameters['password']

    const response = await this.request({
      path: `/sso/login`,
      method: 'POST',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async login(requestParameters: SsoLoginRequest, initOverrides?: RequestInit): Promise<SsoLoginResponse> {
    const response = await this.loginRaw(requestParameters, initOverrides)
    const result = await response.value()
    return result.data as SsoLoginResponse
  }

  async registerRaw(requestParameters: SsoRegisterRequest, initOverrides?: RequestInit) {
    if (requestParameters['username'] == null) {
      throw new Error('Required parameter "username" was null or undefined')
    }
    if (requestParameters['password'] == null) {
      throw new Error('Required parameter "password" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    queryParameters['username'] = requestParameters['username']
    queryParameters['password'] = requestParameters['password']
    if (requestParameters['telephone'] != null) {
      queryParameters['telephone'] = requestParameters['telephone']
    }
    if (requestParameters['authCode'] != null) {
      queryParameters['authCode'] = requestParameters['authCode']
    }

    const response = await this.request({
      path: `/sso/register`,
      method: 'POST',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async register(requestParameters: SsoRegisterRequest, initOverrides?: RequestInit): Promise<ApiResponseVoid> {
    const response = await this.registerRaw(requestParameters, initOverrides)
    return await response.value()
  }

  async infoRaw(requestParameters: SsoInfoRequest, initOverrides?: RequestInit) {
    if (requestParameters['xUserId'] == null) {
      throw new Error('Required parameter "xUserId" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    headerParameters['X-User-Id'] = String(requestParameters['xUserId'])

    const response = await this.request({
      path: `/sso/info`,
      method: 'GET',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async info(requestParameters: SsoInfoRequest, initOverrides?: RequestInit): Promise<ApiResponseUserDTO> {
    const response = await this.infoRaw(requestParameters, initOverrides)
    const result = await response.value()
    return result as ApiResponseUserDTO
  }
}

export const ssoApi = new SsoApi(apiConfig)

/**
 * Relying Party ID — 应与后端配置及 iOS associated domains 保持一致。
 * 开发期间可使用 localhost；生产环境改为实际域名（如 "example.com"）。
 */
export const RP_ID: string = process.env.EXPO_PUBLIC_RP_ID ?? 'localhost'
