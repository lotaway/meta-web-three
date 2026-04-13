import type { TurboModule } from 'react-native'
import { TurboModuleRegistry } from 'react-native'

export interface Spec extends TurboModule {
  generateRequestSignature(params: Object, secretKey: string): Promise<string>
  preciseAmountSum(amountA: string, amountB: string): Promise<string>
  computeOrderTotal(unitPrice: string, quantity: number, discountAmount: string, shippingFee: string): Promise<string>
  hmacSign(message: string, signingKey: string): Promise<string>
  createNonce(): Promise<string>
  systemTimestampMs(): Promise<number>

  createPasskey(rpId: string, userName: string): Promise<string>
  getPasskeyList(): Promise<Array<string>>
  authenticatePasskey(challenge: string): Promise<boolean>
  deletePasskey(credentialId: string): Promise<void>
}

export default TurboModuleRegistry.getEnforcing<Spec>('Appsdk');

