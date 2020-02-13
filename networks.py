import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset

from dataset import EnableDataset

import pickle

from sklearn.model_selection import KFold, StratifiedKFold,ShuffleSplit ,train_test_split

import copy
import os
import random

import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
import pdb

class Network(nn.Module):
    def __init__(self,INPUT_NUM,NUMB_CLASS):
        super().__init__()
        self.sclayer1 = nn.Sequential(
            nn.Conv2d(INPUT_NUM, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.sclayer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.drop_out = nn.Dropout()

        # self.fc1 = nn.Linear( 4096, 2000)
        self.fc1 = nn.Linear( 6144, 2000)
        self.fc2 = nn.Linear(2000, NUMB_CLASS)


    def forward(self,x):
        x = self.sclayer1(x)
        x = self.sclayer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Network_modespecific(nn.Module):
    def __init__(self,INPUT_NUM,NUMB_CLASS):
        super().__init__()
        self.sclayer1 = nn.Sequential(
            nn.Conv2d(INPUT_NUM, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.sclayer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear( 4101, 2000)
        self.fc2 = nn.Linear(2000, NUMB_CLASS)


    def forward(self,x,y):
        x = self.sclayer1(x)
        x = self.sclayer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = torch.cat((x,y),1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class MyResNet18(ResNet):

    def __init__(self):
        super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        
    def forward(self, x):
        # change forward here
        # pdb.set_trace()
        x1=x[0]
        y=x[1]
        
        
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1= self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x1 = self.avgpool(x1)
        x1 = torch.flatten(x1, 1)
        # pdb.set_trace()
        x1 = torch.cat((x1,y),1)
        x1= self.fc(x1)

        return x1



