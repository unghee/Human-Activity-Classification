import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from scipy import signal, stats
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np


class EnableDataset(Dataset):
    def __init__(self, file_path, window_size=1000, stride=500, delay=500, processed=False):
        raw_data = pd.read_csv(file_path)
        if processed:
            segmented_data, self.labels = self.segment_processed_data(raw_data, window_size=window_size)
        else:
            segmented_data, self.labels = self.segment_unprocessed_data(raw_data, window_size=window_size, stride=stride,delay=delay)
        self.img_data = self.spectrogram2(segmented_data)

    def __len__(self):
        return self.img_data.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.img_data[idx]), self.labels[idx]

    # Variable time step implementation, returns list of numpy arrays
    def segment_processed_data_2(self, raw_data, delay=500):
        segmented_data = []
        labels = []
        timesteps = []
        index = 0
        prev_end = -1
        while not pd.isnull(raw_data.loc[index, 'Right_Heel_Contact']) and not pd.isnull(raw_data.loc[index, 'Left_Heel_Contact']):
            right_heel = raw_data.loc[index, 'Right_Heel_Contact']
            left_heel = raw_data.loc[index, 'Left_Heel_Contact']
            if (prev_end != -1):
                timesteps.append((prev_end, min(right_heel, left_heel)))
            timesteps.append((min(right_heel, left_heel), max(right_heel, left_heel)))
            prev_end = max(right_heel, left_heel)
            index += 1

        for timestep in timesteps:
            if (timestep[1] + delay < raw_data.shape[0]):
                labels.append(raw_data.loc[timestep[1]+delay, 'Mode'])
                segmented_data.append(raw_data.loc[timestep[0]:timestep[1]-1, 'Right_Shank_Ax':'Left_Knee'])
        return segmented_data, labels

    # Get window_size rows before right and left heel contacts. The labels are the labels at the right and left heel contact times
    def segment_processed_data(self, raw_data, window_size=1000):
        segmented_data = np.array([], dtype=np.int64).reshape(0,window_size,48)
        labels = np.array([], dtype=np.int64)
        timesteps = []
        index = 0
        while not pd.isnull(raw_data.loc[index, 'Right_Heel_Contact']):
            timesteps.append(raw_data.loc[index, 'Right_Heel_Contact'])
            index += 1
        index = 0
        while not pd.isnull(raw_data.loc[index, 'Left_Heel_Contact']):
            timesteps.append(raw_data.loc[index, 'Left_Heel_Contact'])
            index += 1

        for timestep in timesteps:
            if timestep-window_size-1 >= 0:
                labels = np.append(labels, [raw_data.loc[timestep, 'Mode']], axis=0)
                data = np.expand_dims(raw_data.loc[timestep-window_size-1:timestep-2, 'Right_Shank_Ax':'Left_Knee'], axis=0)
                segmented_data = np.concatenate((segmented_data, data), axis=0)
        return segmented_data, labels



    # Returns segmented_data, labels
    # segmented_data is numpy array of dimension: (<number of segments> X <window size> X <number of columns>)
    # labels is numpy array of dimension : (<number of segments> X ,) and each label is the label <delay> rows ahead
    # in the raw_data

    # raw_data : pandas dataframe object, it is the data to segment
    # window_size : number of rows in each segment,
    # stride : how far to move the window by
    # delay : the number of samples to look forward in the dataset to get the label
    def segment_unprocessed_data(self, raw_data, window_size=1000, stride=500, delay=500):
        segmented_data = np.array([], dtype=np.int64).reshape(0,window_size,48)
        labels = np.array([], dtype=np.int64)
        for i in range(0, raw_data.shape[0] - window_size, stride):
            labels = np.append(labels, [raw_data.loc[i+delay, 'Mode']], axis=0)
            data = np.expand_dims(raw_data.loc[i:i+window_size-1, 'Right_Shank_Ax':'Left_Knee'], axis=0)
            segmented_data = np.concatenate((segmented_data, data), axis=0)
        return segmented_data, labels

    def spectrogram(self, segmented_data, fs=500):
        ret = []
        for y in range(segmented_data.shape[0]):
            vals1 = []
            for x in range(3):
                row = segmented_data[y,:,x]
                f, t, Sxx = signal.spectrogram(row, fs, window=signal.windows.hamming(100, True), noverlap=50)
                fig = plt.figure()
                ax = fig.add_axes([0.,0.,1.,1.])
                fig.set_size_inches((5,5))
                ax.pcolormesh(t, f, Sxx, cmap='gray')
                ax.axis('off')
                fig.add_axes(ax)
                fig.canvas.draw()
                # this rasterized the figure
                X = np.array(fig.canvas.renderer._renderer)
                X = 0.2989*X[:,:,1] + 0.5870*X[:,:,2] + 0.1140*X[:,:,3]
                vals1.append(X)
                plt.close()
            vals2 = []
            for x in range(6,9):
                row = segmented_data[y,:,x]
                f, t, Sxx = signal.spectrogram(row, fs, window=signal.windows.hamming(100, True), noverlap=50)
                fig = plt.figure()
                ax = fig.add_axes([0.,0.,1.,1.])
                fig.set_size_inches((5,5))
                ax.pcolormesh(t, f, Sxx, cmap='gray')
                ax.axis('off')
                fig.add_axes(ax)
                fig.canvas.draw()
                # this rasterized the figure
                X = np.array(fig.canvas.renderer._renderer)
                X = 0.2989*X[:,:,1] + 0.5870*X[:,:,2] + 0.1140*X[:,:,3]
                vals2.append(X)
                plt.close()

            out1 = np.stack(vals1, axis=2).astype(np.uint8)
            out2 = np.stack(vals2, axis=2).astype(np.uint8)
            out = np.hstack((out1, out2))
            ret.append(out)
            cv2.imshow("ret", out)
            cv2.waitKey(0)
        ret = np.stack(ret)
        # print(ret.shape)
        return ret

    def spectrogram2(self, segmented_data, fs=500):
        ret = []
        for y in range(segmented_data.shape[0]):
            vals1 = []
            for x in range(3):
                row = segmented_data[y,:,x]
                f, t, Sxx = signal.spectrogram(row, fs, window=signal.windows.hamming(100, True), noverlap=50)
                tmp, _ = stats.boxcox(Sxx.reshape(-1,1))
                Sxx = tmp.reshape(Sxx.shape)-np.min(tmp)
                Sxx = Sxx/np.max(Sxx)*255
                vals1.append(Sxx)
            vals2 = []
            for x in range(6,9):
                row = segmented_data[y,:,x]
                f, t, Sxx = signal.spectrogram(row, fs, window=signal.windows.hamming(100, True), noverlap=50)
                tmp, _ = stats.boxcox(Sxx.reshape(-1,1))
                Sxx = tmp.reshape(Sxx.shape)-np.min(tmp)
                Sxx = Sxx/np.max(Sxx)*255
                vals2.append(Sxx)

            out1 = np.stack(vals1, axis=2)
            out2 = np.stack(vals2, axis=2)
            out = np.hstack((out1, out2))
            #out = 20*np.log10(np.abs(out)/1000000000000000000)
            #out = out-np.min(out)
            #out = np.flipud(out/np.max(out)*255)
            #print(np.min(out), np.max(out))
            out = np.flipud(out)
            out=np.asarray(out).transpose(2,1,0)/128.0-1.0
            ret.append(out.astype(np.uint8))
            #plt.imshow(out[:,:,0].astype(np.uint8))
            #plt.show()
            # ret = np.asarray(ret).transpose(0, 2, 3, 1)/128.0-1.0
        ret = np.stack(ret)
        # ret = np.asarray(ret).transpose(0, 2, 3, 1)/128.0-1.0
        # ret = np.asarray(ret).astype("f").transpose(2, 0, 1)/128.0-1.0
        #print(ret.shape)
        return ret
