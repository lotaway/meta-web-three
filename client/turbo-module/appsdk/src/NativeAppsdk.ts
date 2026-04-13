import type { TurboModule } from 'react-native'
import { TurboModuleRegistry } from 'react-native'

export interface Spec extends TurboModule {
  generateRequestSignature(params: Object, secretKey: string): Promise<string>
  preciseAmountSum(amountA: string, amountB: string): Promise<string>
  computeOrderTotal(unitPrice: string, quantity: number, discountAmount: string, shippingFee: string): Promise<string>
  hmacSign(message: string, signingKey: string): Promise<string>
  createNonce(): Promise<string>
  systemTimestampMs(): Promise<number>

  // Passkey methods
  createPasskey(rpId: string, userName: string): Promise<string> // 返回 credentialId
  getPasskeyList(): Promise<Array<string>> // 列出可用 Passkeys
  authenticatePasskey(challenge: string): Promise<boolean> // 验证，使用 challenge 防重放
  deletePasskey(credentialId: string): Promise<void>
}

export default TurboModuleRegistry.getEnforcing<Spec>('Appsdk');

