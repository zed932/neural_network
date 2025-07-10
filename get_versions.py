import torch, tensorflow, onnx

print(f"PyTorch: {torch.__version__}, GPU: {torch.backends.mps.is_available()}")
print(f"TensorFlow: {tensorflow.__version__}, GPU: {tensorflow.config.list_physical_devices('GPU')}")
print(f"ONNX: {onnx.__version__}")