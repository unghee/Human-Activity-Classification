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


import sys,os
sys.path.append('.')

from utils import *
from networks import *

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
NUMB_CLASS = 5
NUB_EPOCH= 200
numfolds = 3
DATA_LOAD_BOOL = True

SAVING_BOOL = True

MODE = 'bilateral'
CLASSIFIER = 'LDA'
SENSOR = ["imu","emg","goin"]
sensor_str='_'.join(SENSOR)

from itertools import combinations
BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 51),bands=10,hop_length=10,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE)
#BIO_train= EnableDataset(subject_list= ['156'],data_range=(1, 4),bands=10,hop_length=10,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE)

save_object(BIO_train,'count_Data.pkl')

# with open('count_Data.pkl', 'rb') as input:
# 	   BIO_train = pickle.load(input)

# vals = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
vals = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
for img, labels, trigger in BIO_train:
    # vals[trigger.astype(int)][labels.astype(int)]+=1
    vals[int(trigger)][int(labels)]+=1

print(vals)