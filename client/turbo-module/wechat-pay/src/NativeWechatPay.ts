import type { TurboModule } from 'react-native'
import { TurboModuleRegistry } from 'react-native'

export interface WechatPayParams {
    partnerId: string
    prepayId: string
    nonceStr: string
    timeStamp: string
    packageValue: string
    sign: string
}

export interface Spec extends TurboModule {
    init(appId: string): void
    isWechatInstalled(): Promise<boolean>
    pay(params: WechatPayParams): Promise<void>
    emitFinalConfirmEvent(message: string): void
    addListener(eventName: string): void
    removeListeners(count: number): void
}

export default TurboModuleRegistry.getEnforcing<Spec>('WechatPay')
