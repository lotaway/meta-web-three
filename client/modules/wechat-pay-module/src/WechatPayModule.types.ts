export interface WechatPayParams {
  partnerId: string;
  prepayId: string;
  nonceStr: string;
  timeStamp: string;
  packageValue: string;
  sign: string;
}

export interface WechatPayEvent {
  type: 'payment_success' | 'payment_error' | 'payment_cancelled';
  message?: string;
  data?: any;
}

export type WechatPayListener = (event: WechatPayEvent) => void;
