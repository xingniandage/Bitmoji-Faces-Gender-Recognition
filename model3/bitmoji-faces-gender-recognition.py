import os
import sys

import torch
import time
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import csv
#############<<  AlexNet  >>########

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
#################################################
class MyDataset(Dataset):
    def __init__(self, transforms=None, isTrain=None, isVal = None):
        super(MyDataset, self).__init__()
        self.transforms = transforms
        self.isTrain = isTrain
        images = []
        labels = []
        if self.isTrain:
            data = pd.read_csv('/kaggle/input/bitmoji-faces-gender-recognition/train.csv')
            path = '/kaggle/input/bitmoji-faces-gender-recognition/BitmojiDataset/trainimages/'
            y = data['is_male'].tolist()
            x = data['image_id'].tolist()
            for i in range(0, len(x)):
                if isVal:
                    if i < 2400:
                        continue
                else:
                    if i == 2400:
                        break
                images.append(path + x[i])
                l = 1 if y[i] == 1 else 0
                labels.append(l)

            self.labels = labels
        else:
            path = '/kaggle/input/bitmoji-faces-gender-recognition/BitmojiDataset/testimages/'
            for i in range(3000, 4084):
                images.append(path + str(i) + '.jpg')
        self.images = images

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = np.array(image, np.float32) / 255.0

        if self.isTrain:
            label = self.labels[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.images)


def train():
    batch_size = 30
    transform = transforms.Compose([transforms.RandomResizedCrop(224),# 随机裁剪，再缩放成 224×224
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform1 = transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = MyDataset(transforms=transform, isTrain=True, isVal=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = MyDataset(transforms=transform1,isTrain=True,isVal=True)
    validate_loader = DataLoader(val_set, batch_size=12, shuffle=True)

    model = AlexNet()
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    train_steps = len(train_loader)
    val_num = len(val_set)
    best_acc = 0.0
    epoch = 0
    print("start training")
    while epoch < 10:
        running_loss = 0.0
        time_start = time.perf_counter()  # 对训练一个 epoch 计时
        train_bar = tqdm(train_loader, file=sys.stdout)  # 对训练一个 epoch 计时
        #for i, data in enumerate(train_loader):
        for step, data in enumerate(train_bar):  # 遍历训练集，step从0开始计算
            inputs, labels = data[0], data[1]
            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 打印训练进度（使训练过程可视化）
            rate = (step + 1) / len(train_loader)  # 当前进度 = 当前step / 训练一轮epoch所需总step
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
            print()
            print('%f s' % (time.perf_counter() - time_start))

        # validate
        model.eval()  # 验证过程中关闭 Dropout
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images)
                predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                (epoch + 1, running_loss / train_steps, val_accurate))
        # 保存准确率最高的那次网络参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), './log.pth')

        epoch += 1
    print('Finish training')


def inference():
    print('Start inference...')
    batch_size = 30
    transform = transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = MyDataset(transforms=transform, isTrain=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model = AlexNet()
    model.eval()
    checkpoint = torch.load('./log.pth')
    model.load_state_dict(checkpoint)
    result = []
    for i, data in enumerate(test_loader):
        outputs = model(data)
        idx = outputs.argmax(axis=1)
        for j in idx:
            if j == 1:
                result.append(1)
            else:
                result.append(-1)

    data = []
    count = 3000
    header = ['image_id', 'is_male']
    for i in result:
        row = []
        row.append(str(count) + '.jpg')
        row.append(i)
        data.append(row)
        count = count + 1
    with open("./sumbmission.csv", mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print('Start finish.')

train()
inference()
