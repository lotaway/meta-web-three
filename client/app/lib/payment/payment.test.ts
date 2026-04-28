import { describe, it, expect, vi, beforeEach } from 'vitest'
import { pay, type PayResult } from '../app/lib/payment'

vi.mock('@app/wechat-pay', () => ({
  NativeWechatPay: {
    init: vi.fn(),
    pay: vi.fn(),
    isWechatInstalled: vi.fn().mockResolvedValue(true),
  },
}))

vi.mock('@app/alipay', () => ({
  NativeAlipay: {
    init: vi.fn(),
    pay: vi.fn().mockResolvedValue('transaction_id'),
  },
}))

vi.mock('@stripe/stripe-react-native', () => ({
  confirmPayment: vi.fn(),
}))

describe('payment module', () => {
  describe('wechatPay', () => {
    it('should return success when wechat pay succeeds', async () => {
      const result = await pay('wechat', {
        partnerId: 'partner123',
        prepayId: 'prepay123',
        nonceStr: 'nonce',
        timeStamp: '1234567890',
        packageValue: 'Sign=WXPay',
        sign: 'sign',
      })
      expect(result.status).toBe('success')
    })
  })

  describe('alipayPay', () => {
    it('should return success when alipay succeeds', async () => {
      const result = await pay('alipay', 'order_string')
      expect(result.status).toBe('success')
    })
  })

  describe('stripePay', () => {
    it('should return fail when clientSecret missing', async () => {
      const result = await pay('stripe', {
        clientSecret: '',
        returnURL: 'app://payment',
      })
      expect(result.status).toBe('fail')
    })
  })
})