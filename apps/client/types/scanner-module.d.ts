declare module 'scanner-module' {
  import type { ComponentType } from 'react'

  export type ScannerModuleViewProps = {
    isScanning?: boolean
    onBarcodeScanned?: (event: { nativeEvent?: { url?: string }; url?: string }) => void
    onScanSuccess?: (event: { nativeEvent: { data: string } }) => void
    onError?: (event: { nativeEvent: { message: string } }) => void
    style?: unknown
  }

  export const ScannerModuleView: ComponentType<ScannerModuleViewProps>
  export function requestCameraPermissionAsync(): Promise<boolean>
  export function scanFromURLAsync(url: string): Promise<string>

  const ScannerModule: {
    requestCameraPermissionAsync: typeof requestCameraPermissionAsync
    scanFromURLAsync: typeof scanFromURLAsync
  }

  export default ScannerModule
}
