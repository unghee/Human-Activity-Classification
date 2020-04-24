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
        self.fc1 = nn.Linear(6149, 2000)
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


class LSTM(nn.Module):
    def __init__(self, n_channels, n_hidden=256, n_layers=2, n_classes=5, drop_prob=0.5,gpubool=False):
        #https://github.com/dspanah/Sensor-Based-Human-Activity-Recognition-LSTMsEnsemble-Pytorch/blob/master/notebooks/1.0-dsp-LSTMsEnsemle.ipynb
        super(LSTM, self).__init__()
        
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_channels = n_channels
        self.gpubool=gpubool
        
        self.lstm  = nn.LSTM(n_channels, n_hidden, n_layers, dropout=self.drop_prob,batch_first=True)
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)

        
    def forward(self, x, hidden,batch_size):
        
        # x = x.permute(1, 0, 2)
        # batch_size = x.size(0)
        x = x.permute(0, 2, 1).float()
        x = x.float()
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)    
        x = x.contiguous().view(-1, self.n_hidden)
        out = self.fc(x)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (self.gpubool):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda().float(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda().float())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().float(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().float())

        return hidden   


def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)