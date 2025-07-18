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

# Стиль графиков
plt.style.use("ggplot")

#Сиды для одинаковых результатов работы с нейросетью
seed = 50
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Выбор железа для обучения нейронной сети
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Загружаем модель
num_classes = 5
model = models.mobilenet_v3_small(num_classes)

#Определяем пути к датасету
train_data_dir = '../dataset/train'
val_data_dir = '../dataset/validation'

#Определяем трансформацию изображений
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#Создаём датасеты
train_dataset = ImageFolder(train_data_dir, transform = train_transforms)
val_dataset = ImageFolder(val_data_dir, transform = val_transforms)

#Создаём датагенератор
batch_size = 20
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

#Посчитаем количество данных в датасетах
train_class_counts = np.zeros(len(train_dataset.classes))
for _, label in train_dataset:
    train_class_counts[label] += 1

val_class_counts = np.zeros(len(val_dataset.classes))
for _, label in val_dataset:
    val_class_counts[label] += 1

# #Создаём графику
# fig, ax = plt.subplots(1, 2, figsize=(14,6))
#
# #Barplot для Train
# sns.barplot(x = train_dataset.classes, y = train_class_counts, ax = ax[0])
# ax[0].set_title('Train dataset')
# ax[0].set_xlabel('Class')
# ax[0].set_ylabel('Number of images')
# ax[0].tick_params(axis = 'x', rotation = 90)
#
# #Barplot для Validation
# sns.barplot(x = val_dataset.classes, y = val_class_counts, ax = ax[1])
# ax[1].set_title('Validation dataset')
# ax[1].set_xlabel('Class')
# ax[1].set_ylabel('Number of images')
# ax[1].tick_params(axis = 'x', rotation = 90)
#
# plt.tight_layout()
# "plt.show()"

#Обучение Нейронной сетки
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)

#Число эпох
num_epochs = 30

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_accuracy = 0
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    #Валидация модели
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}],'
          f'Train loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:4f}')

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(),  'best_model.pth')
        print("Saved best model!")

    torch.save(model.state_dict(), 'last_model.pth')
    print()

print('Training and validation complete')


