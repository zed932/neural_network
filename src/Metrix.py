import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from PIL import Image

from Metrix_func import evaluate_model, plot_confusion_matrix

# Стиль графиков
plt.style.use("ggplot")

#Выбор железа для обучения нейронной сети
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Определяем пути к датасету
val_data_dir = '../dataset/validation'

#Определяем трансформацию изображений
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#Создаём датасеты
val_dataset = ImageFolder(val_data_dir, transform = val_transforms)

#Создаём датагенератор(Батчи)
batch_size = 20
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

#Загружаем модель c нашими весами
num_classes = 5
model = models.mobilenet_v3_small(num_classes)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)

#Оценка модели и построение матрицы ошибок
cm, report, accuracy_1, weighted_f1_1 = evaluate_model(model, val_loader)
print("Metrics for current model:")
print(report)
print(f'Test Accuracy: {accuracy_1:.4f}')
plot_confusion_matrix(cm, classes = list(range(5)))
