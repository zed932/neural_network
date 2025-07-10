import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def create_model(num_classes=10):
    model = mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(1024, num_classes)  # Замена последнего слоя
    return model