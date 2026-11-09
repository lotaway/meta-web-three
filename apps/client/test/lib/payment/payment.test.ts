import { pay } from '../../../src/lib/payment'

jest.mock('wechat-pay-module', () => {
  const mock = {
    init: jest.fn(),
    pay: jest.fn().mockResolvedValue(undefined),
    isWechatInstalled: jest.fn().mockResolvedValue(true),
  }
  return {
    __esModule: true,
    default: mock,
    WechatPayModule: mock,
  }
})

jest.mock('alipay-module', () => {
  const mock = {
    init: jest.fn(),
    pay: jest.fn().mockResolvedValue('transaction_id'),
  }
  return {
    __esModule: true,
    default: mock,
    AlipayModule: mock,
  }
})

jest.mock('@/api/generated', () => ({
  ALIPAY_APP_ID: 'alipay-test-app-id',
  WECHAT_APP_ID: 'wechat-test-app-id',
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
      expect(result).toMatchObject({ type: 'wechat', transactionId: 'prepay123' })
    })
  })

  describe('alipayPay', () => {
    it('should return success when alipay succeeds', async () => {
      const result = await pay('alipay', 'order_string')
      expect(result.status).toBe('success')
      expect(result).toMatchObject({ type: 'alipay', transactionId: 'transaction_id' })
    })
  })

  describe('stripePay', () => {
    it('should return fail when clientSecret missing', async () => {
      const result = await pay('stripe', {
        clientSecret: '',
        returnURL: 'app://payment',
      })
      expect(result.status).toBe('fail')
      expect(result).toMatchObject({ type: 'stripe', code: 'invalid_client_secret' })
    })
  })
})
