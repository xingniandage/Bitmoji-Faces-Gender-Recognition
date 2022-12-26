# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset
from model import MyCNN
from torch.utils.tensorboard import SummaryWriter
from AverageMeter import AverageMeter


# topk的准确率计算
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True)  # 使用topk来获得前k个的索引
    pred = pred.t()  # 进行转置
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred))  # 与正确标签序列形成的矩阵相比，生成True/False矩阵
    #     print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)  # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size))  # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn


def train(path_dir,label_dir, batch_size, workers, EPOCH, tensorboard_path,
          model_cp,transform=None):
    # load data
    dataset = MyDataset(path_dir=path_dir, label_dir=label_dir, transform=transform)
    # 用PyTorch的DataLoader类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=workers)

    # train model
    cnn = MyCNN()
    # cuda
    # if torch.cuda.is_available() is True:
    #     print('Cuda is available!')
    #     cnn = cnn.cuda
    # train model
    cnn.train()

    # 指定优化器
    optimizer = torch.optim.Adagrad(cnn.parameters(), lr=1e-3)
    # 指定loss函数
    loss_func = nn.CrossEntropyLoss()
    train_loss = 0.0
    top1 = AverageMeter()
    tj = 0
    for epo in range(EPOCH):
        cnt = 0
        for img, label in dataloader:  # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
            img, label = Variable(img), Variable(label.long())  # 将数据放置在PyTorch的Variable节点中
            out = cnn(img)[0]  # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
            loss = loss_func(out,
                             label.squeeze())  # 计算损失，也就是网络输出值和实际label的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维Tensor
            loss.backward()  # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
            optimizer.step()  # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.zero_grad()  # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加
            prec1, prec2 = accuracy(out, label, topk=(1, 2))
            n = img.size(0)
            top1.update(prec1.item(), n)
            train_loss += loss.item()
            print('Frame: ', cnt, ' train_loss: %.6f' % (train_loss / (cnt + 1)), ' train_acc: %.6f' % top1.avg)

            # tensorBoard 曲线绘制
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Train/Loss', loss.item(), tj)
            writer.add_scalar('Train/Accuracy', top1.avg, tj)
            writer.flush()
            #
            tj += 1
            cnt += 1
    #
    torch.save(cnn.state_dict(), model_cp)  # 训练所有数据后，保存网络的参数

