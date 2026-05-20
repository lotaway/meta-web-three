import type { TurboModule } from 'react-native'
import { TurboModuleRegistry } from 'react-native'

export interface Spec extends TurboModule {
  generateRequestSignature(params: Object, secretKey: string): string
  preciseAmountSum(amountA: string, amountB: string): string
  computeOrderTotal(unitPrice: string, quantity: number, discountAmount: string, shippingFee: string): string
  hmacSign(message: string, signingKey: string): string
  createNonce(): string
  systemTimestampMs(): number

  createPasskey(rpId: string, userName: string): Promise<string>
  getPasskeyList(): Array<string>
  authenticatePasskey(rpId: string, challenge: string): Promise<string>
  deletePasskey(credentialId: string): boolean
}

export default TurboModuleRegistry.getEnforcing<Spec>('Appsdk');

