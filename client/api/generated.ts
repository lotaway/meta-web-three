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

  async getAuthCodeRaw(requestParameters: { telephone: string }, initOverrides?: RequestInit) {
    if (requestParameters['telephone'] == null) {
      throw new Error('Required parameter "telephone" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    queryParameters['telephone'] = requestParameters['telephone']

    const response = await this.request({
      path: `/sso/getAuthCode`,
      method: 'GET',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async getAuthCode(requestParameters: { telephone: string }, initOverrides?: RequestInit) {
    const response = await this.getAuthCodeRaw(requestParameters, initOverrides)
    return await response.value()
  }

  async loginByPhoneRaw(requestParameters: { telephone: string; authCode: string }, initOverrides?: RequestInit) {
    if (requestParameters['telephone'] == null) {
      throw new Error('Required parameter "telephone" was null or undefined')
    }
    if (requestParameters['authCode'] == null) {
      throw new Error('Required parameter "authCode" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    queryParameters['telephone'] = requestParameters['telephone']
    queryParameters['authCode'] = requestParameters['authCode']

    const response = await this.request({
      path: `/sso/loginByPhone`,
      method: 'POST',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async loginByPhone(requestParameters: { telephone: string; authCode: string }, initOverrides?: RequestInit): Promise<SsoLoginResponse> {
    const response = await this.loginByPhoneRaw(requestParameters, initOverrides)
    const result = await response.value()
    return result.data as SsoLoginResponse
  }

  async updatePasswordRaw(requestParameters: { telephone: string; password: string; authCode: string }, initOverrides?: RequestInit) {
    if (requestParameters['telephone'] == null) {
      throw new Error('Required parameter "telephone" was null or undefined')
    }
    if (requestParameters['password'] == null) {
      throw new Error('Required parameter "password" was null or undefined')
    }
    if (requestParameters['authCode'] == null) {
      throw new Error('Required parameter "authCode" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    queryParameters['telephone'] = requestParameters['telephone']
    queryParameters['password'] = requestParameters['password']
    queryParameters['authCode'] = requestParameters['authCode']

    const response = await this.request({
      path: `/sso/updatePassword`,
      method: 'POST',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async updatePassword(requestParameters: { telephone: string; password: string; authCode: string }, initOverrides?: RequestInit): Promise<ApiResponseVoid> {
    const response = await this.updatePasswordRaw(requestParameters, initOverrides)
    return await response.value()
  }
}

export const ssoApi = new SsoApi(apiConfig)

export interface ProductCollectionAddRequest {
  productId: number
  productName: string
  productPic: string
  productPrice?: number
  productSubTitle?: string
}

export interface ProductCollectionDetailRequest {
  xUserId: number
  productId: number
}

export interface ProductCollectionDeleteRequest {
  xUserId: number
  productId: number
}

export interface ProductCollectionListRequest {
  xUserId: number
  pageNum?: number
  pageSize?: number
}

export class ProductCollectionApi extends BaseAPI {
  async addRaw(requestParameters: ProductCollectionAddRequest, initOverrides?: RequestInit) {
    if (requestParameters['productId'] == null) {
      throw new Error('Required parameter "productId" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    const response = await this.request({
      path: `/member/productCollection/add`,
      method: 'POST',
      headers: headerParameters,
      query: queryParameters,
      body: requestParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async add(requestParameters: ProductCollectionAddRequest, initOverrides?: RequestInit): Promise<ApiResponseVoid> {
    const response = await this.addRaw(requestParameters, initOverrides)
    return await response.value()
  }

  async detailRaw(requestParameters: ProductCollectionDetailRequest, initOverrides?: RequestInit) {
    if (requestParameters['xUserId'] == null) {
      throw new Error('Required parameter "xUserId" was null or undefined')
    }
    if (requestParameters['productId'] == null) {
      throw new Error('Required parameter "productId" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    headerParameters['X-User-Id'] = String(requestParameters['xUserId'])
    queryParameters['productId'] = requestParameters['productId']

    const response = await this.request({
      path: `/member/productCollection/detail`,
      method: 'GET',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async detail(requestParameters: ProductCollectionDetailRequest, initOverrides?: RequestInit) {
    const response = await this.detailRaw(requestParameters, initOverrides)
    return await response.value()
  }

  async deleteRaw(requestParameters: ProductCollectionDeleteRequest, initOverrides?: RequestInit) {
    if (requestParameters['xUserId'] == null) {
      throw new Error('Required parameter "xUserId" was null or undefined')
    }
    if (requestParameters['productId'] == null) {
      throw new Error('Required parameter "productId" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    headerParameters['X-User-Id'] = String(requestParameters['xUserId'])
    queryParameters['productId'] = requestParameters['productId']

    const response = await this.request({
      path: `/member/productCollection`,
      method: 'DELETE',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async delete(requestParameters: ProductCollectionDeleteRequest, initOverrides?: RequestInit): Promise<ApiResponseVoid> {
    const response = await this.deleteRaw(requestParameters, initOverrides)
    return await response.value()
  }

  async listRaw(requestParameters: ProductCollectionListRequest, initOverrides?: RequestInit) {
    if (requestParameters['xUserId'] == null) {
      throw new Error('Required parameter "xUserId" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    headerParameters['X-User-Id'] = String(requestParameters['xUserId'])
    if (requestParameters['pageNum'] != null) {
      queryParameters['pageNum'] = requestParameters['pageNum']
    }
    if (requestParameters['pageSize'] != null) {
      queryParameters['pageSize'] = requestParameters['pageSize']
    }

    const response = await this.request({
      path: `/member/productCollection/list`,
      method: 'GET',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async list(requestParameters: ProductCollectionListRequest, initOverrides?: RequestInit) {
    const response = await this.listRaw(requestParameters, initOverrides)
    return await response.value()
  }
}

export const productCollectionApi = new ProductCollectionApi(apiConfig)

export interface CouponListRequest {
  xUserId: number
  useStatus?: number
}

export interface CouponClaimRequest {
  xUserId: number
  couponTypeId: number
}

export class CouponApi extends BaseAPI {
  async listRaw(requestParameters: CouponListRequest, initOverrides?: RequestInit) {
    if (requestParameters['xUserId'] == null) {
      throw new Error('Required parameter "xUserId" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    headerParameters['X-User-Id'] = String(requestParameters['xUserId'])
    if (requestParameters['useStatus'] != null) {
      queryParameters['useStatus'] = requestParameters['useStatus']
    }

    const response = await this.request({
      path: `/member/coupon/coupons`,
      method: 'GET',
      headers: headerParameters,
      query: queryParameters,
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async list(requestParameters: CouponListRequest, initOverrides?: RequestInit) {
    const response = await this.listRaw(requestParameters, initOverrides)
    return await response.value()
  }

  async claimRaw(requestParameters: CouponClaimRequest, initOverrides?: RequestInit) {
    if (requestParameters['xUserId'] == null) {
      throw new Error('Required parameter "xUserId" was null or undefined')
    }

    const queryParameters: QueryParameter = {}
    const headerParameters: HTTPHeaders = {}

    headerParameters['X-User-Id'] = String(requestParameters['xUserId'])

    const response = await this.request({
      path: `/member/coupon/coupons/claim`,
      method: 'POST',
      headers: headerParameters,
      query: queryParameters,
      body: { couponTypeId: requestParameters.couponTypeId },
    }, initOverrides)

    return new runtime.JSONApiResponse<any>(response)
  }

  async claim(requestParameters: CouponClaimRequest, initOverrides?: RequestInit): Promise<ApiResponseVoid> {
    const response = await this.claimRaw(requestParameters, initOverrides)
    return await response.value()
  }
}

export const couponApi = new CouponApi(apiConfig)

/**
 * Relying Party ID — 应与后端配置及 iOS associated domains 保持一致。
 * 开发期间可使用 localhost；生产环境改为实际域名（如 "example.com"）。
 */
export const RP_ID: string = process.env.EXPO_PUBLIC_RP_ID ?? 'localhost'
