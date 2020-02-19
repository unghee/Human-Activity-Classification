import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm # Displays a progress bar
plt.rcParams["font.family"] = "Times New Roman"
csfont = {'fontname':'Times New Roman'}
matplotlib.rcParams.update({'font.size': 16})
#del matplotlib.font_manager.weight_dict['roman']
#matplotlib.font_manager._rebuild()
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
sensors=["imu","emg","goin"]
MODE = ['bilateral']
#BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=BAND,hop_length=HOP)
#BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 51),bands=BAND,hop_length=HOP,model_type="CNN",sensors=sensors,mode=MODE)
BIO_train= EnableDataset(subject_list= ['156'],data_range=(1, 20),bands=BAND,hop_length=HOP,model_type="CNN",sensors=sensors,mode=MODE)


train_size = int(0.8 * len(BIO_train))
test_size = int((len(BIO_train) - train_size))
train_dataset, test_dataset= torch.utils.data.random_split(BIO_train, [train_size, test_size])
# Create dataloaders
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# classes = [0,0,0,0,0,0,0]
# for data, labels in trainloader:
#     for x in range(labels.size()[0]):
#         classes[labels[x]] +=1
# print(classes)
valloader = DataLoader(test_dataset, batch_size=1)
testloader = DataLoader(test_dataset, batch_size=1)


# numb_class = 5


device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())
model = Network(52, 5)
model.load_state_dict(torch.load('/home/justin/Documents/Human-Activity-Classification/Freq-Encoding/bestmodel_BATCH_SIZE32_LR1e-05_WD0.001_EPOCH200.pth', map_location='cpu'))
model = model.to(device)
mask = np.load("/home/justin/Documents/Human-Activity-Classification/mask.npy")

# weights = torch.FloatTensor([0.0, 1.0, 9693/2609, 9693/3250, 9693/1181, 9693/1133, 9693/530 ])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_epoch = NUB_EPOCH

inputimg = [None]*128
argmax = [0]*128
loc = [None]*128

def evaluate(model, loader):
    model.eval()
    correct = 0

    for batch in loader:
        batch = batch[0]
        batch = batch.to(device)
        x, xtemp = model.forward(batch)
        xtemp = xtemp.data.cpu().numpy()
        for i in range(128):
            img = xtemp[0,i,:,:]
            max = np.max(img)
            if(max >= argmax[i]):
                loc[i] = np.unravel_index(img.argmax(), img.shape)
                inputimg[i] = batch.cpu().numpy()[0,:,:,:]
                argmax[i] = max




evaluate(model, testloader)

print(mask.shape, inputimg[4].shape)

fig, ax = plt.subplots(4)
fig.tight_layout()
fig.text(0.5, 0.04, 'Time (s)', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
fig.text(0.04, 0.5, 'Frequency (Hz)', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize']) 
im = ax[0].imshow(np.flipud(inputimg[5][2,:,0:50]).astype(float))
ax[0].set_xticks([0,12.5, 25, 37.5, 49])
ax[0].set_xticklabels([0,0.25, 0.5, 0.75, 1])
#ax[0].set_xticks([0, 25, 49])
#ax[0].set_xticklabels([0, 0.5, 1])
#ax[0].set_xlabel("Time (s)")
ax[0].set_yticks([0,4,8])
ax[0].set_yticklabels([250, 150,50])
#ax[0].set_yticks([0, 2, 4, 6, 8])
#ax[0].set_yticklabels([250, 200, 150, 100, 50])
#ax[0].set_ylabel("Hz")
#ax[0].title.set_text("Right Shank Ax IMU Spectrogram")
ax[0].title.set_text("IMU")
cbar = plt.colorbar(im, ax=ax[0:3])
cbar.ax.set_ylabel("Magnitude (dB)", rotation=270, labelpad=20)
ax[1].imshow(np.flipud(inputimg[5][35,:,0:50]).astype(float))
ax[1].set_xticks([0,12.5, 25, 37.5, 49])
ax[1].set_xticklabels([0,0.25, 0.5, 0.75, 1])
#ax[1].set_xticks([0, 25, 49])
#ax[1].set_xticklabels([0, 0.5, 1])
#ax[1].set_xlabel("Time (s)")
ax[1].set_yticks([0,4,8])
ax[1].set_yticklabels([250, 150,50])
#ax[1].set_yticks([0, 2, 4, 6, 8])
#ax[1].set_yticklabels([250, 200, 150, 100, 50])
#ax[1].set_ylabel("Hz")
#ax[1].title.set_text("Right Vastus Lateralis EMG Spectrogram")
ax[1].title.set_text("EMG")
ax[2].imshow(np.flipud(inputimg[5][46,:,0:50]).astype(float))
ax[2].set_xticks([0,12.5, 25, 37.5, 49])
ax[2].set_xticklabels([0,0.25, 0.5, 0.75, 1])
#ax[2].set_xticks([0, 25, 49])
#ax[2].set_xticklabels([0, 0.5, 1])
#ax[2].set_xlabel("Time (s)")
ax[2].set_yticks([0,4,8])
ax[2].set_yticklabels([250, 150,50])
#ax[0].set_yticks([0, 2, 4, 6, 8])
#ax[0].set_yticklabels([250, 200, 150, 100, 50])
#ax[2].set_ylabel("Hz")
#ax[2].title.set_text("Right Knee Goniometer Spectrogram")
ax[2].title.set_text("Goniometer")

out = np.zeros_like(mask[:,:,0, 0])
for i in range(128):
    argval = loc[i]
    if argval is None:
        continue
    out += mask[:,:,argval[0], argval[1]]
    
out = np.flipud(out/np.max(out))

im2 = ax[3].imshow(out, cmap="plasma")
ax[3].set_xticks([0,12.5, 25, 37.5, 49])
ax[3].set_xticklabels([0,0.25, 0.5, 0.75, 1])
#ax[3].set_xticks([0, 25, 49])
#ax[3].set_xticklabels([0, 0.5, 1])
#ax[3].set_xlabel("Time (s)")
ax[3].set_yticks([0,4,8])
ax[3].set_yticklabels([250, 150,50])
#ax[0].set_yticks([0, 2, 4, 6, 8])
#ax[0].set_yticklabels([250, 200, 150, 100, 50])
#ax[3].set_ylabel("Hz")
ax[3].title.set_text("Activation Map")
cbar2 = plt.colorbar(im2, ax=ax[3], aspect = 4)
cbar2.ax.set_ylabel("Magnitude", rotation=270, labelpad = 33)

plt.show()