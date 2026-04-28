import { registerWebModule, NativeModule } from 'expo';

import { ScannerModuleEvents } from './ScannerModule.types';

class ScannerModule extends NativeModule<ScannerModuleEvents> {
  async requestCameraPermissionAsync(): Promise<boolean> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      stream.getTracks().forEach(track => track.stop());
      return true;
    } catch {
      return false;
    }
  }
}
  }

  startScanning() {}
  stopScanning() {}
}

export default registerWebModule(ScannerModule, 'ScannerModule');