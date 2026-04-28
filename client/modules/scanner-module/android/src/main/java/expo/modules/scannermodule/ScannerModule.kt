package expo.modules.scannermodule

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class ScannerModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("ScannerModule")

    Events("onScanSuccess", "onError")

    Function("requestCameraPermissionAsync") {
      true
    }

    Function("startScanning") {
    }

    Function("stopScanning") {
    }

    View(ScannerModuleView::class) {
      Prop("isScanning") { view: ScannerModuleView, scanning: Boolean ->
        view.setIsScanning(scanning)
      }
      Events("onScanSuccess", "onError")
    }
  }
}