

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import DetectDataSet
from model import MyCNN


def detect(model_path, image_path, data_transform, batch_size, workers):

    dataset = DetectDataSet(image_path, data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers)
    # load model
    cnn = MyCNN()
    result=[]
    # cuda
    if torch.cuda.is_available() is True:
        print('Cuda is available!')
        cnn = cnn.cuda()
    cnn.load_state_dict(torch.load(model_path))
    for img, label in dataloader:  # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
        img = Variable(img)
        out = cnn(img)[0]  # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
        print(out)
        for i in range(len(out)):
            if out[i, 0] > out[i, 1]:  # 猫的概率大于狗
                result.append(f'0')
            else:  # 猫的概率小于狗
                result.append(f'1')
        break
    return result


