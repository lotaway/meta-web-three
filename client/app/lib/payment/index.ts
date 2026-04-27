import { NativeWechatPay, WechatPayParams } from '@app/wechat-pay'
import { NativeAlipay } from '@app/alipay'
import { confirmPayment } from '@stripe/stripe-react-native'
import { ALIPAY_APP_ID, WECHAT_APP_ID } from '@/api/generated'

export type PayType = 'wechat' | 'alipay' | 'stripe'

export interface WechatPayInput extends WechatPayParams {
  type: 'wechat'
}

export interface AlipayInput {
  type: 'alipay'
  orderString: string
}

export interface StripeInput {
  type: 'stripe'
  clientSecret: string
  returnURL: string
}

export type PayInput = WechatPayInput | AlipayInput | StripeInput

export interface PaySuccessResult {
  status: 'success'
  type: PayType
  transactionId?: string
}

export interface PayCancelResult {
  status: 'cancel'
  type: PayType
}

export interface PayFailResult {
  status: 'fail'
  type: PayType
  code?: string
  message?: string
}

export type PayResult = PaySuccessResult | PayCancelResult | PayFailResult

type WechatRuntimeParams = WechatPayParams & { appId?: string }

export async function pay(
  type: PayType,
  params: WechatPayParams | string | { clientSecret: string; returnURL: string }
): Promise<PayResult> {
  switch (type) {
    case 'wechat':
      return wechatPay(params as WechatPayParams)

    case 'alipay':
      return alipayPay(params as string)

    case 'stripe':
      return stripePay(params as { clientSecret: string; returnURL: string })
  }
}

async function wechatPay(
  params: WechatPayParams
): Promise<PaySuccessResult | PayCancelResult | PayFailResult> {
  try {
    const runtimeParams = params as WechatRuntimeParams
    const appId = runtimeParams.appId || WECHAT_APP_ID
    if (!appId) {
      return { status: 'fail', type: 'wechat', code: 'APP_ID_MISSING', message: '微信 App ID 未配置' }
    }
    NativeWechatPay.init(appId)
    await NativeWechatPay.pay(runtimeParams)
    return { status: 'success', type: 'wechat', transactionId: runtimeParams.prepayId }
  } catch (error: any) {
    if (error?.code === 'USER_CANCEL') {
      return { status: 'cancel', type: 'wechat' }
    }
    return {
      status: 'fail',
      type: 'wechat',
      code: error?.code,
      message: error?.message
    }
  }
}

async function alipayPay(
  orderString: string
): Promise<PaySuccessResult | PayCancelResult | PayFailResult> {
  try {
    if (ALIPAY_APP_ID) {
      NativeAlipay.init(ALIPAY_APP_ID)
    }
    const result = await NativeAlipay.pay({ orderString })
    return { status: 'success', type: 'alipay', transactionId: result }
  } catch (error: any) {
    if (error?.code === '6001') {
      return { status: 'cancel', type: 'alipay' }
    }
    return {
      status: 'fail',
      type: 'alipay',
      code: error?.code,
      message: error?.message
    }
  }
}

async function stripePay(
  params: { clientSecret: string; returnURL: string }
): Promise<PaySuccessResult | PayCancelResult | PayFailResult> {
  try {
    const { paymentIntent, error } = await confirmPayment({
      clientSecret: params.clientSecret,
      returnURL: params.returnURL
    })

    if (error) {
      if (error.code === 'cancelled') {
        return { status: 'cancel', type: 'stripe' }
      }
      return {
        status: 'fail',
        type: 'stripe',
        code: error.code,
        message: error.message
      }
    }

    if (paymentIntent?.status === 'succeeded') {
      return { status: 'success', type: 'stripe', transactionId: paymentIntent.id }
    }

    return {
      status: 'fail',
      type: 'stripe',
      code: paymentIntent?.status,
      message: 'Payment not completed'
    }
  } catch (error: any) {
    return {
      status: 'fail',
      type: 'stripe',
      code: error?.code,
      message: error?.message
    }
  }
}
