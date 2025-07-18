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

#Загружаем нашу обученную модель
num_classes = 5
model = models.mobilenet_v3_small(num_classes)

model.load_state_dict(torch.load('best_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

#Предобработка изображения
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#Работа с фотографиями
image_path = '../dataset/test/1.jpg'
image = Image.open(image_path)
image_tensor = preprocess(image)
image_tensor = image_tensor.unsqueeze(0).to(device)

#Прогоняем через модель
with torch.no_grad():
    output = model(image_tensor)

#Применяем softmax для получения вероятностей классов
probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu()

#Отображаем результаты
top_prob, top_class = torch.topk(probabilities, 1)
top_prob = top_prob.item()
top_class = top_class.item()

class_names = ['cat', 'cow', 'dog', 'horse', 'human']
class_name = class_names[top_class]

plt.imshow(image)
plt.axis('off')
plt.title(f'Predcited: {class_name} ({top_prob*100:.2f}%)')
plt.show()
