# 基于CNN的表情性别分类模型
---
##1.环境准备
&emsp;&emsp;需要安装pytorch,pillow,numpy,pandas,如需查看训练精度及损失图则需要安装tensorboard
##2.数据集
&emsp;&emsp;本次训练数据集均来自kaggle。[点此下载](https://www.kaggle.com/competitions/bitmoji-faces-gender-recognition/data)
##3.参数设置
&emsp;&emsp;所用参数均在config.py中设置
##4.卷积网络结构
###第一层 卷积网络 输入：3 输出：16 卷积核：5*5 步长：1 padding：2
```
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )```
###第二层 卷积网络 输入：16 输出：32 卷积核：5*5 padding：2
```
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )```
###第三层 卷积网络 输入：32 输出：64 卷积核：5*5
```  
self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )```
###第四层 卷积网络 输入：32 输出：64 卷积核：5*5
```
self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )```
##5.训练结果
&emsp;&emsp;该模型训练最终准确率为99.3%
&emsp;&emsp;训练结果查看
```tensorboard --logdir =<directory_name > ```
##6.预测结果
&emsp;&emsp;预测结果输出为0与1组成的列表，0代表女性，1代表男性