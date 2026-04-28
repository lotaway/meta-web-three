package expo.modules.scannermodule

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.util.Size
import android.view.View
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ScannerModuleView(
  private val context: Context,
  private val appContext: AppContext
) : ExpoView(context, appContext) {

  private var cameraExecutor: ExecutorService? = null
  private var cameraProvider: ProcessCameraProvider? = null
  private var isScanning = false

  private val onScanSuccess = EventDispatcher<Map<String, Any>>()
  private val onError = EventDispatcher<Map<String, Any>>()

  fun startCamera() {
    if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
      != PackageManager.PERMISSION_GRANTED
    ) {
      onError(mapOf("message" to "Camera permission not granted"))
      return
    }

    if (isScanning) return
    isScanning = true

    cameraExecutor = Executors.newSingleThreadExecutor()

    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    cameraProviderFuture.addListener({
      try {
        cameraProvider = cameraProviderFuture.get()

        val preview = Preview.Builder().build().also {
          it.surfaceProvider = surfaceProvider
        }

        val imageAnalysis = ImageAnalysis.Builder()
          .setTargetResolution(Size(1280, 720))
          .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
          .build()
          .also {
            it.setAnalyzer(cameraExecutor!!) { imageProxy ->
              processImage(imageProxy)
            }
          }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        cameraProvider?.unbindAll()
        cameraProvider?.bindToLifecycle(
          context as androidx.lifecycle.LifecycleOwner,
          cameraSelector,
          preview,
          imageAnalysis
        )
      } catch (e: Exception) {
        onError(mapOf("message" to (e.message ?: "Camera initialization failed")))
      }
    }, ContextCompat.getMainExecutor(context))
  }

  @OptIn(ExperimentalGetImage::class)
  private fun processImage(imageProxy: ImageProxy) {
    val mediaImage = imageProxy.image
    if (mediaImage != null) {
      val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
      val scanner = BarcodeScanning.getClient()

      scanner.process(image)
        .addOnSuccessListener { barcodes ->
          for (barcode in barcodes) {
            if (barcode.valueType == Barcode.TYPE_URL || barcode.valueType == Barcode.TYPE_TEXT) {
              barcode.rawValue?.let { value ->
                onScanSuccess(mapOf("data" to value))
                stopCamera()
                return@addOnSuccessListener
              }
            }
          }
        }
        .addOnFailureListener { e ->
          onError(mapOf("message" to (e.message ?: "Scan failed")))
        }
        .addOnCompleteListener {
          imageProxy.close()
        }
    } else {
      imageProxy.close()
    }
  }

  fun stopCamera() {
    isScanning = false
    cameraProvider?.unbindAll()
    cameraExecutor?.shutdown()
  }

  fun setIsScanning(scanning: Boolean) {
    if (scanning) {
      startCamera()
    } else {
      stopCamera()
    }
  }

  override fun onDetachedFromWindow() {
    super.onDetachedFromWindow()
    stopCamera()
  }
}