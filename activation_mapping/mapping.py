import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset

import sys
sys.path.append('.')
from dataset import EnableDataset
from mask_net import FigureNetwork

#plots the activation figure as seen in the paper

########## SETTINGS  ########################
BAND=10
HOP=10
sensors=["imu","emg","gon"]
MODE = ['bilateral']
############################################

#import any subset of data
BIO_train= EnableDataset(subject_list= ['156'],data_range=(1, 20),bands=BAND,hop_length=HOP,model_type="CNN",sensors=sensors,mode=MODE)
testloader = DataLoader(BIO_train, batch_size=1)

#initialize device
device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())

#initialize network
model = FigureNetwork(52, 5)
#load trained state dict here if you can!
#model.load_state_dict(torch.load(<your state dict here>, map_location='cpu'))
model = model.to(device)

#be sure to generate the mask fist using mask_gen.py
mask = np.load("mask.npy")

inputimg = [None]*128
argmax = [0]*128
loc = [None]*128

#evaluate the model based on a single data 
model.eval()
correct = 0

#find the maximal activations per channel
for batch in testloader:
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

#plot figure
fig, ax = plt.subplots(4)
fig.tight_layout()
fig.text(0.5, 0.04, 'Time (s)', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
fig.text(0.04, 0.5, 'Frequency (Hz)', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize']) 
im = ax[0].imshow(np.flipud(inputimg[5][2,:,0:50]).astype(float))
ax[0].set_xticks([0,12.5, 25, 37.5, 49])
ax[0].set_xticklabels([0,0.25, 0.5, 0.75, 1])
ax[0].set_yticks([0,4,8])
ax[0].set_yticklabels([250, 150,50])
ax[0].title.set_text("IMU")
cbar = plt.colorbar(im, ax=ax[0:3])
cbar.ax.set_ylabel("Magnitude (dB)", rotation=270, labelpad=20)
ax[1].imshow(np.flipud(inputimg[5][35,:,0:50]).astype(float))
ax[1].set_xticks([0,12.5, 25, 37.5, 49])
ax[1].set_xticklabels([0,0.25, 0.5, 0.75, 1])
ax[1].set_yticks([0,4,8])
ax[1].set_yticklabels([250, 150,50])
ax[1].title.set_text("EMG")
ax[2].imshow(np.flipud(inputimg[5][46,:,0:50]).astype(float))
ax[2].set_xticks([0,12.5, 25, 37.5, 49])
ax[2].set_xticklabels([0,0.25, 0.5, 0.75, 1])
ax[2].set_yticks([0,4,8])
ax[2].set_yticklabels([250, 150,50])
ax[2].title.set_text("Goniometer")

#make figure from mask
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
ax[3].set_yticks([0,4,8])
ax[3].set_yticklabels([250, 150,50])
ax[3].title.set_text("Activation Map")
cbar2 = plt.colorbar(im2, ax=ax[3], aspect = 4)
cbar2.ax.set_ylabel("Magnitude", rotation=270, labelpad = 33)

plt.show()