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
import time

import librosa.display
import pdb

class EnableDataset(Dataset):
    '''
    Generated using the ENABL3S dataset found here: https://doi.org/10.6084/m9.figshare.5362627
    Dataset is designed as outlined in: https://ieeexplore.ieee.org/abstract/document/9134897


    dataDir:        path to folder containing data

    subject_list:   the subjects to be included in dataset

    model_type:     configures data to be fed into a type of model.
                        CNN: creates 2D melspectrogram data from the sensor data provided in dataDir
                        Random_modespecific or Random:  uses 1D sensor data directly from dataDir and provides label of previous data point as well.
                        LDA or SVM: for feature-based classifiers. Use 1D sensor data directly from dataDir and will NOT provides label of previous data point.

    sensors:        list of sensors to use in dataset. Must be subset of ["imu","emg", "goin"]
                        imu:  include inertial measurmenet unit data
                        emg:  include electromyography data
                        goin: include goniometer data

    mode:           configures the label for each datum. Must be "ipsilateral", "contralateral" or "bilateral"
                        ipsilateral:   label is the action from the next step of the same foot
                        contralateral: label is the action from the next step of the opposite foot
                        bilateral:     label is the action from the next step (same or opposite foot)


    CNN Parameters
    exclude_list:   data files in dataDir that will be excluded. This is if certain files of specific subjects would like to be excluded but not in others.

    mode_specific:  flag to run general (false) or mode_specific (true) configurations
                        general:        exclude ground-truth, current locomotor activty (previous label) in data point
                        mode_specific:  include the ground-truth current locomotor activty (previous label) in data point

    data_range:     the specified circuit trials for each subject

    window_size:    how many samples to consider for a label

    transform:      optional torch transform object to apply to the data

    bands:          number of frequency bands in spectrogram

    hop_length:     number of samples between sucessive frames in spectrogram


    NN Parameters
    prevlabel: When provided, will only include data that comes after prevlebel

    delay: Specify a delay in 1D signal data

    phaselabel: Will only include data with specifed leg phase value if prevlabel is provided
    '''
    def __init__(self, dataDir='./Data/', subject_list=['156'], model_type="CNN", exclude_list=[], mode_specific=False, data_range=(1, 51), window_size=500, sensors=["imu","emg", "goin"], mode="bilateral", transform=None, bands=None, hop_length=None, phaselabel=None, prevlabel=None, delay=0):
        self.model_type = model_type
        self.dataset = []
        self.prev_label = np.array([], dtype=np.int64)

        if self.model_type == "CNN":
            print("\trange: [%d, %d)" % (data_range[0], data_range[1]))

            self.img_data_stack=np.empty((51, 3, 4, 51), dtype=np.int64)
            self.transform = transform
            self.mode_specific=mode_specific

            self.avgSpectrogramTime = 0.0
            numSpectrogramsProcessed = 0

            for subjects in subject_list:
                for i in range(data_range[0], data_range[1]):
                    filename = dataDir +'AB' + subjects+'/Processed/'+'AB' + subjects+ '_Circuit_%03d_post.csv'% i
                    if not os.path.exists(filename) or ('AB' + subjects+ '_Circuit_%03d_post'% i) in exclude_list:
                        print(filename, 'not found or excluded')
                        continue
                    raw_data = pd.read_csv(filename)
                    segmented_data = np.array([], dtype=np.int64).reshape(0,window_size,48)
                    labels = np.array([], dtype=np.int64)
                    timestep_type = []
                    timesteps = []
                    triggers = []
                    index = 0
                    gait_event_types = []

                    # Find the timesteps at which all gait events occur and the action (label) prior to the event.
                    gait_events = ['Right_Heel_Contact','Right_Toe_Off','Left_Heel_Contact','Left_Toe_Off']
                    for event in gait_events:
                        while not pd.isnull(raw_data.loc[index, event]):
                            trigger = raw_data.loc[index, event+'_Trigger']
                            trigger=str(int(trigger))
                            if float(trigger[2]) != 6 and float(trigger[0]) !=6: # exclude data where subject is resting
                                timesteps.append(raw_data.loc[index, event])
                                trigger = raw_data.loc[index, event+'_Trigger']
                                trigger=str(int(trigger))
                                triggers.append(trigger) # triggers can be used to compare translational and steady-state error

                                labels = np.append(labels,[float(trigger[2])], axis =0)

                                if trigger[0] == trigger[2]:
                                    timestep_type.append(1)
                                else:
                                    timestep_type.append(0)

                                if "right" in event.lower():
                                    gait_event_types.append("Right")
                                else:
                                    gait_event_types.append("Left")

                                self.prev_label = np.append(self.prev_label,[float(trigger[0])], axis =0)
                            index += 1
                        index = 0

                    # Take raw data at each timesetps collected above, filter it according to the given sensor list and mode and create the melspectrogram.
                    for idx,timestep in enumerate(timesteps):
                        data = raw_data.loc[timestep-window_size-1:timestep-2,:]
                        if timestep-window_size-1 >= 0:
                            if mode == "ipsilateral":
                                data = data.filter(regex='(?=.*'+ gait_event_types[idx] + '|Mode|Waist)(?!.*Toe)(?!.*Heel)(.+)', axis=1)
                            elif mode == "contralateral":
                                opposite = "Left" if gait_event_types[idx] == "Right" else "Right"
                                data = data.filter(regex='(?=.*'+ opposite + '|Mode|Waist)(?!.*Toe)(?!.*Heel)(.+)', axis=1)
                            else:
                                data = data.filter(regex="^((?!Heel|Toe).)*$", axis=1)

                            regex = "(?=!Mode"
                            if "imu" in sensors:
                                regex += "|.*A[xyz].*"
                            if "goin" in sensors:
                                regex += "|.*G[xyz].*|.*Ankle.*|.*Knee.*"
                            if "emg" in sensors:
                                regex += "|.*TA.*|.*MG.*|.*SOL.*|.*BF.*|.*ST.*|.*VL.*|.*RF.*"
                            regex += ")"
                            data = data.filter(regex=regex, axis=1)

                            # Process data into melspectrogram
                            data = np.array(data)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            beg = int(round(time.time()*1000))
                            img= self.melspectrogram(data,bands=bands ,hop_length=hop_length)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            end = int(round(time.time()*1000))
                            self.avgSpectrogramTime += (end - beg) / len(img)
                            numSpectrogramsProcessed += 1

                            if self.mode_specific:
                                self.dataset.append((img,labels[idx],timestep_type[idx],int(self.prev_label[idx])))
                            else:
                                self.dataset.append((img,labels[idx], timestep_type[idx]))
        else:
            for subjects in subject_list:
                    filename = dataDir +'AB' + subjects+'/Features/'+'AB' + subjects+ '_Features_'+ str(300-delay) + '.csv'
                    if not os.path.exists(filename):
                        print(filename, 'not found')
                        continue
                    raw_data = pd.read_csv(filename)

                    timesteps = []
                    timestep_type = []
                    triggers = []
                    index = 0

                    for index in range(0,raw_data.shape[0]):
                        # Get time of each gait event
                        trigger = raw_data.loc[index,'Trigger']
                        trigger=str(int(trigger))
                        phase = raw_data.loc[index,'Leg Phase']
                        if trigger[0] == trigger[2]:
                            timestep_type.append(1)
                        else:
                            timestep_type.append(0)

                        # Filter by prevlabel, phaselabel and sensor and mode parameters
                        if prevlabel is not None:
                            if float(phase) == phaselabel and float(trigger[0]) == prevlabel and float(trigger[2]) != 6 and float(trigger[0]) !=6:
                                triggers.append(trigger)
                                label = float(trigger[2])
                                if float(trigger[2]) == 6:
                                    print('***********',trigger[2])
                                data = raw_data.loc[index, :'Contra RF AR6']
                                if mode == "ipsilateral":
                                    data = data.filter(regex='(?=.*Ipsi.*|.*Waist.*)', axis=0)
                                elif mode == "contralateral":
                                    data = data.filter(regex='(?=.*Contra.*|.*Waist.*)', axis=0)

                                regex = "(?=!Mode|.*Ankle.*|.*Knee.*"
                                if "imu" in sensors:
                                    regex += "|.*A[xyz].*"
                                if "goin" in sensors:
                                    regex += "|.*G[xyz].*|.*Ankle.*|.*Knee.*"
                                if "emg" in sensors:
                                    regex += "|.*TA.*|.*MG.*|.*SOL.*|.*BF.*|.*ST.*|.*VL.*|.*RF.*"
                                regex += ")"
                                data = data.filter(regex=regex, axis=0)
                                data = np.array(data)

                                self.dataset.append((data.T,label, timestep_type[-1]))

                        else:
                            # Filter by sensor and mode parameters
                            if float(trigger[2]) != 6 and float(trigger[0]) !=6:

                                triggers.append(trigger) # triggers can be used to compare translational and steady-state error

                                label = float(trigger[2])
                                if float(trigger[2]) == 0 or float(trigger[0])== 0 :
                                    print('***********',trigger[2])

                                data = raw_data.loc[index, :'Contra RF AR6']

                                if mode == "ipsilateral":
                                    data = data.filter(regex='(?=.*Ipsi.*|.*Waist.*)', axis=0)
                                elif mode == "contralateral":
                                    data = data.filter(regex='(?=.*Contra.*|.*Waist.*)', axis=0)

                                regex = "(?=!Mode|.*Ankle.*|.*Knee.*"
                                if "imu" in sensors:
                                    regex += "|.*A[xyz].*"
                                if "goin" in sensors:
                                    regex += "|.*G[xyz].*|.*Ankle.*|.*Knee.*"
                                if "emg" in sensors:
                                    regex += "|.*TA.*|.*MG.*|.*SOL.*|.*BF.*|.*ST.*|.*VL.*|.*RF.*"
                                regex += ")"
                                data = data.filter(regex=regex, axis=0)
                                data = np.array(data)

                                self.prev_label = np.append(self.prev_label,[float(trigger[0])], axis =0)

                                self.dataset.append((data.T,label, timestep_type[-1]))

        self.avgSpectrogramTime = self.avgSpectrogramTime / numSpectrogramsProcessed


    '''
    Return number of data points in dataset
    '''
    def __len__(self):
        return len(self.dataset)

    '''
    Return a data point based on the given index

    Based on settings from init, will return some or all of the following:
        2D spectrogram (if model is CNN) or 1D signal
        label: ground truth activity
        timestep type: 1 if current activty is same as next activity. 0 if they differ. Used for transitional vs steady-state accuracy analysis
        label of previous data point (for Random_modespecific & Random models or when mode_specific flag is true)
    '''
    def __getitem__(self, index):
        if self.model_type == "CNN":
            if self.mode_specific:
                img, label, timestep_type, prev__label= self.dataset[index]
                if self.transform:
                    img = F.to_pil_image(np.uint8(img))
                    img = self.transform(img)
                    img = np.array(img)
                return torch.FloatTensor(img), torch.LongTensor(np.array(label)), timestep_type, prev__label

            else:
                img, label, timestep_type = self.dataset[index]
                if self.transform:
                    img = F.to_pil_image(np.uint8(img))
                    img = self.transform(img)
                    img = np.array(img)
                return torch.FloatTensor(img), torch.LongTensor(np.array(label)), timestep_type
        else:
            img, label, timestep_type = self.dataset[index]
            if self.model_type== "Random_modespecific" or self.model_type=="Random":
                return img, np.array(label), self.prev_label[index], timestep_type
            else:
                return img, np.array(label), timestep_type

    '''
    Compute melspectrograms for given data.

    segmented_data: pandas DataFrame where each row corresponds to data sample

    fs:             sampling frequency

    bands:          number of spectrogram bands (height of spectrogram)

    hop_length:     number of samples between sucessive frames
    '''
    def melspectrogram(self, segmented_data, fs=500,bands=64 ,hop_length=50):
        vals = []
        for x in range(0,np.shape(segmented_data)[1]):
                row = segmented_data[:,x]
                melspec_full = librosa.feature.melspectrogram(y=row,sr=fs,n_fft=hop_length*2, hop_length=hop_length,n_mels=bands)
                logspec_full = librosa.amplitude_to_db(melspec_full)
                vals.append(logspec_full)
        return vals

    '''
    Output dataset statistics
    Feel free to add other statistic variables
    '''
    def __str__(self):
        return """ EnablesDataset Statistics
                   Model Type: {}

                   Dataset size: {}
                   Average spectrogram processing time: {}
               """.format(self.model_type, len(self), self.avgSpectrogramTime if self.model_type == "CNN" else "N/A")

