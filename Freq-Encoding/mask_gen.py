import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset



import pickle

from sklearn.model_selection import KFold, StratifiedKFold,ShuffleSplit ,train_test_split
from sklearn.metrics import confusion_matrix

from PIL import Image


import copy
import os
import random


import sys,os
sys.path.append('.')
# sys.path.append('../')
from utils import *
from networks import *
from dataset import EnableDataset

from itertools import combinations


########## SETTINGS  ########################

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
NUMB_CLASS = 5
NUB_EPOCH= 200
numfolds = 10
DATA_LOAD_BOOL = True
BAND=10
HOP=10
# BAND=16,HOP=27
SAVING_BOOL = False
############################################



#BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 51),bands=BAND,hop_length=HOP,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE)


device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())

model = Network(52,NUMB_CLASS)
#model.load_state_dict(torch.load("/home/justin/Documents/Human-Activity-Classification/Freq-Encoding/bestmodel_BATCH_SIZE32_LR1e-05_WD0.001_EPOCH200.pth"))
model = model.to(device)

mask = np.zeros((10,50,5,25))

for i in range(10):
    for j in range(50):
        x = np.zeros((1, 52,10,50)).astype(float)
        #x[:,:,i,j] = np.nan
        x[:,:,i,j] = np.nan
        x = torch.from_numpy(x)
        x = x.to(device, dtype=torch.float)
        xtemp = model.forward(x)
        for k in range(5):
            for l in range (25):
                if not np.isfinite(xtemp[0][0][k][l]):
                    mask[i][j][k][l] = 1

print(mask)

np.save("mask", mask)
fig, ax = plt.subplots(nrows=4, ncols=24)
x = 0
y = 0

'''for x in range(4):
    for y in range(24):
        plt.imshow(mask[:,:,x,y])
        plt.show()'''



'''for row in ax:
    y = 0
    for col in row:
        col.imshow(mask[:,:,x,y])
        y +=1
    x+=1
    print(x,y)

plt.show()'''


