import { NativeModule, requireNativeModule } from 'expo';

import { ScannerModuleEvents } from './ScannerModule.types';

declare class ScannerModule extends NativeModule<ScannerModuleEvents> {
  requestCameraPermissionAsync(): Promise<boolean>;
  startScanning(): void;
  stopScanning(): void;
}

export default requireNativeModule<ScannerModule>('ScannerModule');
