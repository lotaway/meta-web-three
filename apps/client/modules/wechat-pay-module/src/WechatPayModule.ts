import { requireNativeModule } from 'expo';

export interface WechatPayParams {
  partnerId: string;
  prepayId: string;
  nonceStr: string;
  timeStamp: string;
  packageValue: string;
  sign: string;
}

export interface WechatPayModuleType {
  init(appId: string): void;
  isWechatInstalled(): Promise<boolean>;
  pay(params: WechatPayParams): Promise<void>;
  emitFinalConfirmEvent(message: string): void;
}

// This will load the native module on iOS and Android.
// On web, it will throw an error since the module is not available.
export const WechatPayModule = requireNativeModule<WechatPayModuleType>('WechatPayModule');

export default WechatPayModule;
