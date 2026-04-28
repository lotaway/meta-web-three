package expo.modules.scannermodule

import android.Manifest
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class ScannerModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("ScannerModule")

    Events("onScanSuccess", "onError")

    AsyncFunction("requestCameraPermissionAsync") { promise ->
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
      val permissions = appContext.legacyModule<expo.modules.interfaces.permissions.Permissions>()
      if (permissions == null) {
        promise.reject("PERMISSIONS_NOT_AVAILABLE", "Permissions module not available")
        return@AsyncFunction
      }
      permissions.askForPermissionsAsync(
        object : expo.modules.core.Promise {
          override fun resolve(res: Any?) {
            val response = res as? Map<String, Any>
            val granted = response?.get(Manifest.permission.CAMERA) as? String
            promise.resolve(granted == "granted")
          }
          override fun reject(code: String, message: String?, throwable: Throwable?) {
            promise.reject(code, message, throwable)
          }
        },
        Manifest.permission.CAMERA
      )
    }

    View(ScannerModuleView::class) {
      Prop("isScanning") { view: ScannerModuleView, scanning: Boolean ->
        view.setIsScanning(scanning)
      }
      Events("onScanSuccess", "onError")
    }
  }
}