import { requireNativeModule } from 'expo'

export interface AlipayParams {
    orderString: string
}

export interface AlipayModuleType {
    init(appId: string): void
    pay(params: AlipayParams): Promise<string>
}

export const AlipayModule = requireNativeModule<AlipayModuleType>('AlipayModule')

export default AlipayModule
