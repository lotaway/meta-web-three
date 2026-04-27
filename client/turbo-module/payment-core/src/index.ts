import { NativeWechatPay, WechatPayParams } from '@app/wechat-pay'
import { NativeAlipay, AlipayParams } from '@app/alipay'

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

export async function pay(
  type: PayType,
  params: WechatPayParams | string
): Promise<PayResult> {
  switch (type) {
    case 'wechat':
      return wechatPay(params as WechatPayParams)

    case 'alipay':
      return alipayPay(params as string)

    case 'stripe':
      return Promise.reject(new Error('Stripe not implemented yet'))
  }
}

async function wechatPay(
  params: WechatPayParams
): Promise<PaySuccessResult | PayCancelResult | PayFailResult> {
  try {
    await NativeWechatPay.pay(params)
    return { status: 'success', type: 'wechat' }
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