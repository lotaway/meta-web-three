package expo.modules.scannermodule

import android.Manifest
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.Promise

class ScannerModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("ScannerModule")

    Events("onScanSuccess", "onError")

    AsyncFunction("requestCameraPermissionAsync") { promise: Promise ->
      val context = appContext.reactContext
      if (context == null) {
        promise.resolve(false)
        return@AsyncFunction
      }
      val result = androidx.core.content.ContextCompat.checkSelfPermission(
        context,
        Manifest.permission.CAMERA
      )
      if (result == android.content.pm.PackageManager.PERMISSION_GRANTED) {
        promise.resolve(true)
        return@AsyncFunction
      }
      promise.resolve(false)
    }

    View(ScannerModuleView::class) {
      Prop("isScanning") { view: ScannerModuleView, scanning: Boolean ->
        view.setIsScanning(scanning)
      }
      Events("onScanSuccess", "onError")
    }
  }
}
