import torch
from model import create_model
import tensorflow as tf

def export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=10).to(device)
    model.load_state_dict(torch.load("outputs/model.pth"))
    model.eval()

    # Экспорт в ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, "outputs/model.onnx")

    # Конвертация в TFLite (через ONNX)
    converter = tf.lite.TFLiteConverter.from_onnx_model("outputs/model.onnx")
    tflite_model = converter.convert()
    with open("outputs/model.tflite", "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    export()