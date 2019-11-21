import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
import numpy as np

class EnableDataset(Dataset):
    def __init__(self, file_path, transform=None):
        raw_data = pd.read_csv(file_path)
        segmented_data, labels = self.segment_data(raw_data)
        self.img_data = None # TODO: Convert raw numerical data to spectrogram images
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        # TODO Make this load the img from disk
        return self.img_data[idx]

    # Returns segmented_data, labels
    # segmented_data is numpy array of dimension: (<number of segments> X <window size> X <number of columns>)
    # labels is numpy array of dimension : (<number of segments> X ,) and each label is the label <delay> rows ahead
    # in the raw_data

    # raw_data : pandas dataframe object, it is the data to segment
    # window_size : number of rows in each segment,
    # stride : how far to move the window by
    # delay : the number of samples to look forward in the dataset to get the label
    def segment_data(self, raw_data, window_size=1000, stride=500, delay=500):
        segmented_data = np.array([], dtype=np.int64).reshape(0,1000,48)
        labels = np.array([], dtype=np.int64)
        for i in range(0, raw_data.shape[0] - window_size, stride):
            labels = np.append(labels, [raw_data.loc[i+delay, 'Mode']], axis=0)
            data = np.expand_dims(raw_data.loc[i:i+window_size-1, 'Right_Shank_Ax':'Left_Knee'], axis=0)
            segmented_data = np.concatenate((segmented_data, data), axis=0)
        return segmented_data, labels

dataset = EnableDataset('AB156_Circuit_001_raw.csv')