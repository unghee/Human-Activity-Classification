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

########## SETTINGS  ########################

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
NUMB_CLASS = 5
NUB_EPOCH= 3
numfolds = 3
DATA_LOAD_BOOL = True


############################################

MODEL_NAME = './models/Freq-Encoding/bestmodel'+ \
        		'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'.pth'

RESULT_NAME= './results/Freq-Encoding/accuracy'+ \
        		'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'.txt'

if not os.path.exists('./models/Freq-Encoding'):
	os.makedirs('./models/Freq-Encoding')

if not os.path.exists('./results/Freq-Encoding'):
	os.makedirs('./results/Freq-Encoding')


# Load the dataset and train, val, test splits
print("Loading datasets...")


device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())
model = Network(NUMB_CLASS)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_epoch = NUB_EPOCH

init_state = copy.deepcopy(model.state_dict())
init_state_opt = copy.deepcopy(optimizer.state_dict())


# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=16,hop_length=27)
BIO_train= EnableDataset(subject_list= ['156'],data_range=(1, 2),bands=16,hop_length=27,model_type='CNN')
# with open('BIO_train_melspectro_500s_bands_16_hop_length_27.pkl', 'rb') as input:
#     BIO_train = pickle.load(input)


wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))


for batch, label, dtype in tqdm(wholeloader,disable=DATA_LOAD_BOOL):
	X = batch
	y = label
	types = dtype

accuracies =[]


skf = KFold(n_splits = numfolds, shuffle = True)
i = 0


train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME)

for train_index, test_index in skf.split(X, y, types):

	model.load_state_dict(init_state)
	optimizer.load_state_dict(init_state_opt)

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	types_train, types_test = types[train_index], types[test_index]

	train_dataset = TensorDataset( X_train, y_train, types_train)
	test_dataset = TensorDataset( X_test, y_test, types_test)

	trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

	print("######################Fold:{}#####################3".format(i+1))
	train_class.train(trainloader,num_epoch)

	model.load_state_dict(torch.load(MODEL_NAME))

	# print("Evaluate on test set")
	accs=train_class.evaluate(testloader)
	accuracies.append(accs)

	i +=1

print('saved on the results')


with open(RESULT_NAME, 'w') as f:
	for item in accuracies:
		f.write("%s\n" % item)
f.close()