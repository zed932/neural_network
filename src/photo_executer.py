import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, DirCreatedEvent, FileCreatedEvent

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
model = models.mobilenet_v3_small(pretrained = True)
model.classifier[3] = torch.nn.Linear(
    in_features=model.classifier[3].in_features,
    out_features = num_classes
)

model.load_state_dict(torch.load('best_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

#Предобработка изображения
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#Создаём наблюдатель, который будет отслеживать появление новых файлов в папке

folder_path = '../server/received_photos'
class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                     os.path.isfile(os.path.join(folder_path, f))]
            images = sorted(files, key=os.path.getmtime)
            image_path = images[-1]
            image = Image.open(image_path)
            image_tensor = preprocess(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Прогоняем через модель
            with torch.no_grad():
                output = model(image_tensor)

            # Применяем softmax для получения вероятностей классов
            probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu()

            # Получаем вероятность верного определения и название класса
            top_prob, top_class = torch.topk(probabilities, 1)
            top_prob = top_prob.item()
            top_class = top_class.item()
            # Логика для записи результата в result.txt
            class_names = ['cat', 'cow', 'dog', 'horse', 'human']
            class_name = class_names[top_class]
            if class_name == 'human' and top_prob > 0.8:
                with open('../server/result.txt', 'w') as f:
                        f.write('true')
                        print('work')
                        f.close()
            else:
                with open('../server/result.txt', 'w') as f:
                    f.write('false')
                    print('dont work')
                    f.close()

event_handler = Handler()
observer = Observer()
observer.schedule(event_handler, folder_path, recursive=True)
observer.start()
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    observer.stop()
observer.join()