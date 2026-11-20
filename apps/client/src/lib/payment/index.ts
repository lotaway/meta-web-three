import WechatPayModule, { type WechatPayParams } from 'wechat-pay-module'
import AlipayModule from 'alipay-module'
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

interface StripePaymentOutcome {
  id?: string
  status?: string
}

interface StripeConfirmErrorOutcome {
  code?: string
  message?: string
}

function payFail(type: PayType, code?: string, message?: string): PayFailResult {
  return { status: 'fail', type, code, message }
}

function payCancel(type: PayType): PayCancelResult {
  return { status: 'cancel', type }
}

function toStripeOutcome(
  paymentIntent: StripePaymentOutcome | undefined,
  error: StripeConfirmErrorOutcome | undefined
): PayResult {
  if (error) {
    if (error.code === 'cancelled') return payCancel('stripe')
    return payFail('stripe', error.code, error.message)
  }
  if (paymentIntent?.status === 'succeeded') {
    return { status: 'success', type: 'stripe', transactionId: paymentIntent.id }
  }
  return payFail('stripe', paymentIntent?.status, 'Payment not completed')
}

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
      return payFail('wechat', 'APP_ID_MISSING', '微信 App ID 未配置')
    }
    WechatPayModule.init(appId)
    await WechatPayModule.pay(runtimeParams)
    return { status: 'success', type: 'wechat', transactionId: runtimeParams.prepayId }
  } catch (error: any) {
    if (error?.code === 'USER_CANCEL') return payCancel('wechat')
    return payFail('wechat', error?.code, error?.message)
  }
}

async function alipayPay(
  orderString: string
): Promise<PaySuccessResult | PayCancelResult | PayFailResult> {
  try {
    if (ALIPAY_APP_ID) {
      AlipayModule.init(ALIPAY_APP_ID)
    }
    const result = await AlipayModule.pay({ orderString })
    return { status: 'success', type: 'alipay', transactionId: result }
  } catch (error: any) {
    if (error?.code === '6001') return payCancel('alipay')
    return payFail('alipay', error?.code, error?.message)
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
    return toStripeOutcome(paymentIntent, error)
  } catch (error: any) {
    return payFail('stripe', error?.code, error?.message)
  }
}
