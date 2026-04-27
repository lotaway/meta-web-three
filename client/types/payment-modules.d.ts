declare module '@app/wechat-pay' {
  export interface WechatPayParams {
    appId?: string
    partnerId: string
    prepayId: string
    nonceStr: string
    timeStamp: string
    packageValue: string
    sign: string
  }

  export const NativeWechatPay: {
    init(appId: string): void
    isWechatInstalled(): Promise<boolean>
    pay(params: WechatPayParams): Promise<void>
  }
}

declare module '@app/alipay' {
  export const NativeAlipay: {
    init(appId: string): void
    pay(params: { orderString: string }): Promise<string>
  }
}

declare module '@stripe/stripe-react-native' {
  export const StripeProvider: (props: { publishableKey: string; children: React.ReactNode }) => JSX.Element
  export function confirmPayment(params: {
    clientSecret: string
    returnURL: string
  }): Promise<{
    paymentIntent?: { id?: string; status?: string }
    error?: { code?: string; message?: string }
  }>
}
