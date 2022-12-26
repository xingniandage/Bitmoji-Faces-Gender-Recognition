import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torchvision
from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms, utils
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    def __init__(self, path_dir,label_dir,transform=None):
        self.imgPathArr = []
        self.labelArr = []
        # load csv file
        # 返回的是一个DataFrame数据
        pd_reader = pd.read_csv(label_dir)
        length = len(pd_reader)
        for i in range(length):
            self.imgPathArr.append(f'{path_dir}{os.sep}{pd_reader["image_id"][i]}')
            if pd_reader["is_male"][i] == 1:
                self.labelArr.append(1)
            elif pd_reader["is_male"][i] == -1:
                self.labelArr.append(0)
        self.transforms = transform

    def __getitem__(self, index):
        label = np.array(int(self.labelArr[index]))
        img_path = self.imgPathArr[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgPathArr)





class DetectDataSet(Dataset):

    def __init__(self, filePath, data_transform):
        self.imgPathArr = []
        self.labelArr = []
        if not os.path.exists(path=filePath):
            print("file not found")
        if os.path.isdir(filePath):
            for img in os.listdir(filePath):
                self.imgPathArr.append(filePath+os.sep+img)
                self.labelArr.append(-1)
        elif os.path.isfile(path=filePath):
            self.imgPathArr.append(filePath)
            self.labelArr.append(-1)
        self.transforms = data_transform

    def __getitem__(self, index):
        label = np.array(int(self.labelArr[index]))
        img_path = self.imgPathArr[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgPathArr)


