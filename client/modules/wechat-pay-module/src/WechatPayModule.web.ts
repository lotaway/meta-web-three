// Web platform implementation for WechatPayModule
// This is a stub implementation that throws errors for unsupported operations

import { WechatPayModuleType, WechatPayParams } from './WechatPayModule';

export class WechatPayModuleWeb implements WechatPayModuleType {
  init(appId: string): void {
    console.warn('WechatPayModule is not supported on web platform');
  }

  async isWechatInstalled(): Promise<boolean> {
    console.warn('WechatPayModule is not supported on web platform');
    return false;
  }

  async pay(params: WechatPayParams): Promise<void> {
    throw new Error('WechatPayModule is not supported on web platform');
  }

  emitFinalConfirmEvent(message: string): void {
    console.warn('WechatPayModule is not supported on web platform');
  }
}

export default new WechatPayModuleWeb();
