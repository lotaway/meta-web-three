import { AlipayModuleType, AlipayParams } from './AlipayModule';

export class AlipayModuleWeb implements AlipayModuleType {
  init(appId: string): void {
    console.warn('AlipayModule is not supported on web platform');
  }

  async pay(params: AlipayParams): Promise<string> {
    throw new Error('AlipayModule is not supported on web platform');
  }
}

export default new AlipayModuleWeb();
