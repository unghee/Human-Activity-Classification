import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from scipy import signal, stats
import matplotlib
from matplotlib import pyplot as plt

from tqdm import tqdm # Displays a progress bar

import pandas as pd
import numpy as np

import os
import torchvision.transforms.functional as F


import librosa

import librosa.display
import pdb

class EnableDataset(Dataset):
    '''
    dataDir: path to folder containing data
    subject_list: the subjects to be included in dataset
    data_range: the specified circuit trials for each subject
    window_size: how many samples to consider for a label
    time_series: when true, use time_series method
    label: when specified, the dataset will only contain data with the given label value
    transform: optional transform to apply to the data
    '''
    def __init__(self, dataDir='./Data/' ,subject_list=['156'], phaselabel=None, prevlabel=None, delay=0):

        # print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.dataset = []
        self.prev_label = np.array([], dtype=np.int64)

        for subjects in subject_list:
                filename = dataDir +'AB' + subjects+'/Features/'+'AB' + subjects+ '_Features_'+ str(300-delay) + '.csv'
                # print('subject ID:' + subjects)
                if not os.path.exists(filename):
                    print(filename, 'not found')
                    continue
                raw_data = pd.read_csv(filename)

                timesteps = []
                triggers = []
                index = 0


                # while not pd.isnull(raw_data.loc[index,'Trigger']):
                for index in range(0,raw_data.shape[0]):
                    trigger = raw_data.loc[index,'Trigger']
                    trigger=str(int(trigger))
                    phase = raw_data.loc[index,'Leg Phase']
                    if prevlabel is not None:
                    	if float(phase) == phaselabel and float(trigger[0]) == prevlabel and float(trigger[2]) != 6 and float(trigger[0]) !=6:
	
	                        triggers.append(trigger) # triggers can be used to compare tr	                        label = float(trigger[2])
	                        if float(trigger[2]) == 6:
	                            print('***********',trigger[2])


	                        data = np.array(raw_data.loc[index, :'Contra RF AR6'])
	                        self.dataset.append((data.T,label))
                    else:
                    	if float(trigger[2]) != 6 and float(trigger[0]) !=6:

	                        triggers.append(trigger) # triggers can be used to compare translational and steady-state error

	                        label = float(trigger[2])
	                        if float(trigger[2]) == 6:
	                            print('***********',trigger[2])

	                        data = np.array(raw_data.loc[index, :'Contra RF AR6'])
	                        self.dataset.append((data.T,label))	                	

        print("load dataset done")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # return torch.FloatTensor(img), torch.LongTensor(np.array(label) )
        return img, np.array(label) 
