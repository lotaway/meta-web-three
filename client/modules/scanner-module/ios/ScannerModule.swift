import ExpoModulesCore

public class ScannerModule: Module {
  public func definition() -> ModuleDefinition {
    Name("ScannerModule")

    Events("onScanSuccess", "onError")

    AsyncFunction("requestCameraPermissionAsync") { () -> Bool in
      return true
    }

    Function("startScanning") {
    }

    Function("stopScanning") {
    }

    View(ScannerModuleView.self) {
      Prop("isScanning") { (view: ScannerModuleView, scanning: Bool) in
        view.setScanning(scanning)
      }
      Events("onScanSuccess", "onError")
    }
  }
}
