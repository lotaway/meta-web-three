import type { TurboModule } from 'react-native'
import { TurboModuleRegistry } from 'react-native'

export interface AlipayParams {
  orderString: string
}

export interface Spec extends TurboModule {
  init(appId: string): void
  pay(params: AlipayParams): Promise<string>
}

export default TurboModuleRegistry.getEnforcing<Spec>('Alipay')