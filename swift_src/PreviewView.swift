//
//  PreviewView.swift
//  CustomCamera
//
//  Created by Сергей Мещеряков on 18.07.2025.
//
import UIKit
import AVFoundation

class PreviewView: UIView {
  
  override class var layerClass: AnyClass {
    return AVCaptureVideoPreviewLayer.self
  }
  
  var videoPreviewLayer: AVCaptureVideoPreviewLayer {
    return layer as! AVCaptureVideoPreviewLayer
  }
}
