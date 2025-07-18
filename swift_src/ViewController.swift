import UIKit
import AVFoundation

class ViewController: UIViewController {
    
    // MARK: - Properties
    private let captureSession = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private let cameraQueue = DispatchQueue(label: "CameraQueue")
    private let photoSettings = AVCapturePhotoSettings()
    
    // UI Elements
    private let previewView = PreviewView()
    private let photoImageView = UIImageView()
    private let shotButton = UIButton()
    private let toggleCameraButton = UIButton() // Добавим кнопку переключения камеры
    
    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        checkCameraPermissions()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewView.frame = view.bounds
        photoImageView.frame = view.bounds
    }
    
    // MARK: - UI Setup
    private func setupUI() {
        view.backgroundColor = .white
        
        // Preview View
        view.addSubview(previewView)
        previewView.videoPreviewLayer.session = captureSession
        previewView.videoPreviewLayer.videoGravity = .resizeAspectFill
        
        // Photo Image View
        view.addSubview(photoImageView)
        photoImageView.contentMode = .scaleAspectFit
        photoImageView.isHidden = true
        
        // Shot Button
        view.addSubview(shotButton)
        shotButton.setImage(UIImage(systemName: "camera"), for: .normal)
        shotButton.tintColor = .white
        shotButton.backgroundColor = .black.withAlphaComponent(0.5)
        shotButton.layer.cornerRadius = 50
        shotButton.addTarget(self, action: #selector(makeShotAction), for: .touchUpInside)
        
        // Layout Button with AutoLayout
        shotButton.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            shotButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            shotButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            shotButton.widthAnchor.constraint(equalToConstant: 100),
            shotButton.heightAnchor.constraint(equalTo: shotButton.widthAnchor)
        ])
    }
    
    // MARK: - Camera Actions
    @objc private func makeShotAction() {
        if photoImageView.isHidden {
            makeShot()
        } else {
            cameraQueue.async {
                self.captureSession.startRunning()
            }
        }
        photoImageView.isHidden.toggle()
    }
    
    private func makeShot() {
      let photoSettings = AVCapturePhotoSettings()
      photoSettings.flashMode = .off
      photoOutput.capturePhoto(with: photoSettings, delegate: self)
  }
    
    // MARK: - Camera Setup
    private func checkCameraPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self = self else { return }
                DispatchQueue.main.async {
                    if granted {
                        self.setupSession()
                    } else {
                        self.showPermissionAlert()
                    }
                }
            }
        case .authorized:
            setupSession()
        case .denied, .restricted:
            showPermissionAlert()
        @unknown default:
            fatalError("Unknown authorization status")
        }
    }
    
    private func setupSession() {
        cameraQueue.async {
            self.captureSession.beginConfiguration()
            
            guard let videoDevice = AVCaptureDevice.default(for: .video) else {
                self.showErrorAlert(message: "No video device available")
                return
            }
            
            do {
                // Input
                let input = try AVCaptureDeviceInput(device: videoDevice)
                if self.captureSession.canAddInput(input) {
                    self.captureSession.addInput(input)
                } else {
                    self.showErrorAlert(message: "Cannot add input to session")
                    return
                }
                
                // Output
                if self.captureSession.canAddOutput(self.photoOutput) {
                    self.captureSession.addOutput(self.photoOutput)
                } else {
                    self.showErrorAlert(message: "Cannot add output to session")
                    return
                }
                
                self.captureSession.commitConfiguration()
                self.captureSession.startRunning()
            } catch {
                self.showErrorAlert(message: error.localizedDescription)
            }
        }
    }
    
    // MARK: - Alerts
    private func showPermissionAlert() {
        let alert = UIAlertController(
            title: "Camera Access Denied",
            message: "Please enable camera access in Settings",
            preferredStyle: .alert
        )
        
        alert.addAction(UIAlertAction(title: "Cancel", style: .default))
        alert.addAction(UIAlertAction(title: "Settings", style: .default) { _ in
            guard let settingsUrl = URL(string: UIApplication.openSettingsURLString) else { return }
            UIApplication.shared.open(settingsUrl)
        })
        
        present(alert, animated: true)
    }
    
    private func showErrorAlert(message: String) {
        DispatchQueue.main.async {
            let alert = UIAlertController(
                title: "Error",
                message: message,
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            self.present(alert, animated: true)
        }
    }
}

// MARK: - AVCapturePhotoCaptureDelegate
extension ViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            showErrorAlert(message: error.localizedDescription)
            return
        }
        
        guard let imageData = photo.fileDataRepresentation(),
              let previewImage = UIImage(data: imageData) else {
            showErrorAlert(message: "Failed to process photo")
            return
        }
        
        DispatchQueue.main.async {
            self.photoImageView.isHidden = false
            self.photoImageView.image = previewImage
        }
        
        // Save to photo library in background
        DispatchQueue.global(qos: .utility).async {
            UIImageWriteToSavedPhotosAlbum(previewImage, nil, nil, nil)
        }
        
        cameraQueue.async {
            self.captureSession.stopRunning()
        }
    }
}
