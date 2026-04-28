import ExpoModulesCore
import AVFoundation

class ScannerModuleView: ExpoView, AVCaptureMetadataOutputObjectsDelegate {
  private var captureSession: AVCaptureSession?
  private var previewLayer: AVCaptureVideoPreviewLayer?
  private var isScanning = false

  let onScanSuccess = EventDispatcher()
  let onError = EventDispatcher()

  required init(appContext: AppContext? = nil) {
    super.init(appContext: appContext)
    clipsToBounds = true
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    previewLayer?.frame = bounds
  }

  func setScanning(_ scanning: Bool) {
    if scanning {
      startCamera()
    } else {
      stopCamera()
    }
  }

  private func startCamera() {
    guard !isScanning else { return }
    isScanning = true

    AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
      guard granted else {
        DispatchQueue.main.async {
          self?.onError(["message": "Camera permission not granted"])
        }
        return
      }
      DispatchQueue.main.async {
        self?.setupCamera()
      }
    }
  }

  private func setupCamera() {
    guard let device = AVCaptureDevice.default(for: .video) else {
      onError(["message": "Camera not available"])
      return
    }

    do {
      let input = try AVCaptureDeviceInput(device: device)
      let session = AVCaptureSession()
      session.addInput(input)

      let output = AVCaptureMetadataOutput()
      session.addOutput(output)
      output.setMetadataObjectsDelegate(self, queue: DispatchQueue.main)
      output.metadataObjectTypes = [.qr, .ean8, .ean13, .code128]

      let previewLayer = AVCaptureVideoPreviewLayer(session: session)
      previewLayer.frame = bounds
      previewLayer.videoGravity = .resizeAspectFill
      layer.addSublayer(previewLayer)

      captureSession = session
      self.previewLayer = previewLayer

      DispatchQueue.global(qos: .userInitiated).async {
        session.startRunning()
      }
    } catch {
      onError(["message": error.localizedDescription])
    }
  }

  private func stopCamera() {
    isScanning = false
    captureSession?.stopRunning()
    captureSession = nil
    previewLayer?.removeFromSuperlayer()
    previewLayer = nil
  }

  func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject], from connection: AVCaptureConnection) {
    guard let metadataObject = metadataObjects.first as? AVMetadataMachineReadableCodeObject,
          let value = metadataObject.stringValue else { return }

    stopCamera()
    onScanSuccess(["data": value])
  }

  override func onDetachedFromWindow() {
    super.onDetachedFromWindow()
    stopCamera()
  }
}