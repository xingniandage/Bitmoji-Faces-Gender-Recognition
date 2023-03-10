import os
import pandas as pd

def make_train_test():
    point=0.8 #训练集和测试集比重

    filename='data/BitmojiDataset/trainimages/'#替换成自己存储训练集的位置
    data = pd.read_csv(r'D:\Word  文档\大三上\深度学习\mycode\data\train.csv',sep=',',header='infer',usecols=[0,1]) #替换成自己存储CSV的位置
    array=data.values[0::,0::]
    print(array.shape)
    for i in range(array.shape[0]):
        if i<point*array.shape[0]:
            with open("train_data.txt", "a") as f:
                if array[i][1]==-1:
                    name = filename + array[i][0] + " " + str(0) + '\n'
                else:
                    name=filename+array[i][0]+" "+str(array[i][1])+'\n'
                f.write(name)
        else:
            with open("test.txt", "a") as f:
                if array[i][1]==-1:
                    name = filename + array[i][0] + " " + str(0) + '\n'
                else:
                    name=filename+array[i][0]+" "+str(array[i][1])+'\n'
                f.write(name)

def make_detection():#制作检验集
    filename = 'data/BitmojiDataset/testimages/'#替换成自己存储检验集的位置 1084张图片那里
    data = pd.read_csv(r'D:\Word  文档\大三上\深度学习\mycode\data\sample_submission.csv', sep=',', header='infer', usecols=[0, 1]) #替换自己提交结果文件的位置
    array = data.values[0::, 0::]
    print(array.shape)
    for i in range(array.shape[0]):
        with open("detect_data.txt", "a") as f:
            if array[i][1] == -1:
                name = filename + array[i][0] + " " + str(0) + '\n'
            else:
                name = filename + array[i][0] + " " + str(array[i][1]) + '\n'
            f.write(name)

make_detection()
