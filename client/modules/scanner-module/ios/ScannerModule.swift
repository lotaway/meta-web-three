import ExpoModulesCore
import AVFoundation

public class ScannerModule: Module {
  public func definition() -> ModuleDefinition {
    Name("ScannerModule")

    Events("onScanSuccess", "onError")

    AsyncFunction("requestCameraPermissionAsync") { () -> Bool in
      let status = AVCaptureDevice.authorizationStatus(for: .video)
      switch status {
      case .authorized:
        return true
      case .notDetermined:
        return await AVCaptureDevice.requestAccess(for: .video)
      default:
        return false
      }
    }

    View(ScannerModuleView.self) {
      Prop("isScanning") { (view: ScannerModuleView, scanning: Bool) in
        view.setScanning(scanning)
      }
      Events("onScanSuccess", "onError")
    }
  }
}
