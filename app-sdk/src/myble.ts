import { TurboModuleRegistry } from 'react-native'
import type { TurboModule } from 'react-native'

export interface Spec extends TurboModule {
  scan(): void
}

export default TurboModuleRegistry.getEnforcing<Spec>('MyBle')