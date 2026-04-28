import { registerWebModule, NativeModule } from 'expo';

import { ScannerModuleEvents } from './ScannerModule.types';

class ScannerModule extends NativeModule<ScannerModuleEvents> {
  PI = Math.PI;
  async setValueAsync(value: string): Promise<void> {
    this.emit('onChange', { value });
  }
  hello() {
    return 'Hello world! 👋';
  }
}

export default registerWebModule(ScannerModule, 'ScannerModule');
