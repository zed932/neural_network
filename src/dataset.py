# dataset.py
import torch
from torchvision import datasets, transforms

def load_data():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.ImageFolder("data/train", transform=transform)
    test_data = datasets.ImageFolder("data/test", transform=transform)
    
    return train_data, test_data