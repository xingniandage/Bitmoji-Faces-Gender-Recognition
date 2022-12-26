# coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
from PIL import Image
import sys

import pandas as pd
import numpy as np
 

#检测指标 检测数据的测试

road='data/BitmojiDataset/testimages'

writepath='data/sample_submission.csv'

epochs = 10 # 训练次数
batch_size = 4  # 批处理大小
num_workers = 0  # 多线程的数目
use_gpu = torch.cuda.is_available()
PATH='./model.pt'
# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# class MyDataset(Dataset):
#     def __init__(self, txt_path, transform = None):
#         fh = open(txt_path, 'r')
#         imgs = []
#         for line in fh:
#             line = line.rstrip()
#             words = line.split()
#             imgs.append((words[0], int(words[1])))
#             self.imgs = imgs
#             self.transform = transform
#     def __getitem__(self, index):
#         fn, label = self.imgs[index]
#         img = Image.open(fn).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label
#     def __len__(self):
#         return len(self.imgs)



# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)



# detect_dataset = MyDataset(txt_path=road, transform=data_transform)
# detect_loader = torch.utils.data.DataLoader(detect_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True,
#                                            num_workers=num_workers)



# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
if(os.path.exists('model.pt')):
    net=torch.load('model.pt')
else:
    print('还没有训练！')
    sys.exit()

if use_gpu:
    net = net.cuda()
print(net)

# # 定义loss和optimizer
# cirterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

def detect():
    outputs=[]
    # 模型测试
    correct = 0
    test_loss = 0.0
    test_total = 0
    test_total = 0
    net.eval()
    result=[]
    flag=0
    for filename in os.listdir(r"./" + road):
        # print(filename) #just for test
        # img is used to store the image data
        images = Image.open(road + "/" + filename)
        images=data_transform(images)
        out = net(images)  # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
        if out[0][0] > out[0][1]:  # 猫的概率大于狗
            result.append(f'-1')
        else:  # 猫的概率小于狗
            result.append(f'1')
        flag+=1


    # for data in detect_loader:
    #     images, labels = data
    #     if use_gpu:
    #         images, labels = Variable(images.cuda()), Variable(labels.cuda())
    #     else:
    #         images, labels = Variable(images), Variable(labels)
    #     out = net(images) # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
    #     _, predicted = torch.max(out.data, 1)
    #
    #
    #     for i in range(len(out)):
    #         if out[i, 0] > out[i, 1]:  # 猫的概率大于狗
    #             result.append(f'0')
    #         else:  # 猫的概率小于狗
    #             result.append(f'1')


    df = pd.read_csv(writepath)
    print(df)
    for i in range(len(result)):
        df.iloc[i,1]=result[i]
    csv_save_path = 'sample_submission.csv'
    df.to_csv(csv_save_path, sep=',', index=False, header=True)
    print(result)

print('detect finish!')


detect()
