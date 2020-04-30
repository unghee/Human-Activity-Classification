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


class LIRLSTM(nn.Module):
    def __init__(self,INPUT_NUM,NUMB_CLASS, n_hidden=256, n_layers=2, n_classes=5, drop_prob=0.5,gpubool=False):
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
        self.drop_prob = drop_prob
        self.gpubool= gpubool
        self.n_layers=n_layers
        self.n_hidden= n_hidden

        # self.fc1 = nn.Linear( 4096, 2000)
        # self.fc1 = nn.Linear( 6144, 2000)
        self.lstm  = nn.LSTM(256, self.n_hidden, self.n_layers, dropout=self.drop_prob,batch_first=True)
        self.fc2 = nn.Linear(n_hidden, NUMB_CLASS)


    def forward(self,x, hidden,batch_size):
        x = self.sclayer1(x)
        x = self.sclayer2(x)
        # x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        # x = self.fc1(x)
        x = x.view(batch_size,x.size(1),-1)
        x = x.permute(0,2,1)
        x, hidden = self.lstm(x)
        x = x.contiguous().view(-1, self.n_hidden)
        x = self.fc2(x)
        x = x.view(batch_size,x.size(1), -1)
        x = x[:,-1,:]

        return x, hidden

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
        x = x.permute(0, 2, 1).float() # ([32, 500, 52])
        num_timestep = x.size(1)
        x = x.float()
        x, hidden = self.lstm(x, hidden) # ([32, 500, 52])
        x = self.dropout(x)    
        x = x.contiguous().view(-1, self.n_hidden)
        out = self.fc(x) #torch.Size([16000, 5])
        # out = out.view(batch_size, -1)
        out = out.view(batch_size,num_timestep, -1)
        # out = out[:,-1]
        out = out[:,-1,:]

        
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

class DeepConvLSTM(nn.Module):
    
    # def __init__(self, n_hidden=128, n_layers=1, n_filters=64, n_classes=5, filter_size=5, drop_prob=0.5):
    def __init__(self, n_channels, n_classes,n_hidden=128, n_layers=1, drop_prob=0.5,gpubool=False,n_filters=64,filter_size=5,SLIDING_WINDOW_LENGTH=500):

        super(DeepConvLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.gpubool=gpubool
        self.n_channels=n_channels
        self.sliding_window_length=SLIDING_WINDOW_LENGTH
             
        self.conv1 = nn.Conv1d(n_channels, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        
        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x, hidden, batch_size):
        
        x = x.view(-1, self.n_channels, self.sliding_window_length)
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(x.size(2), -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)
        
        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = self.fc(x)
        
        out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
        
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
        
    # def init_hidden(self, batch_size):
    #     ''' Initializes hidden state '''
    #     # Create two new tensors with sizes n_layers x batch_size x n_hidden,
    #     # initialized to zero, for hidden state and cell state of LSTM
    #     weight = next(self.parameters()).data
        
    #     if (self.gpubool):
    #         hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
    #               weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
    #     else:
    #         hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
    #                   weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
    #     return hidden




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