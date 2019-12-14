import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from scipy import signal, stats
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from tqdm import tqdm # Displays a progress bar

import pandas as pd
import numpy as np

import os
import torchvision.transforms.functional as F
import pdb

class EnableDataset(Dataset):
    def __init__(self, dataDir='./Data/' ,subject_list=['156'], data_range=(1, 10), window_size=500, stride=500, delay=500, processed=True, label=None, transform=None):

        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.dataset = []
        self.prev_label = np.array([], dtype=np.int64)
        self.img_data_stack=np.empty((51, 3, 4, 51), dtype=np.int64)
        self.transform = transform

        for subjects in subject_list:
            for i in range(data_range[0], data_range[1]):
                filename = dataDir +'AB' + subjects+'/Processed/'+'AB' + subjects+ '_Circuit_%03d_post.csv'% i
                if not os.path.exists(filename):
                    print(filename, 'not found')
                    continue
                raw_data = pd.read_csv(filename)

                segmented_data = np.array([], dtype=np.int64).reshape(0,window_size,48)
                labels = np.array([], dtype=np.int64)
                timesteps = []
                triggers = []
                index = 0

                gait_events = ['Right_Heel_Contact','Right_Toe_Off','Left_Heel_Contact','Left_Toe_Off']
                for event in gait_events:
                    while not pd.isnull(raw_data.loc[index, event]):
                        trigger = raw_data.loc[index, event+'_Trigger']
                        trigger=str(int(trigger))
                        # if label == 2:
                        #     if float(trigger[0]) == label:
                        #         print(label,trigger[0])
                        #         pdb.set_trace()
                        if label is None or float(trigger[0]) == label:
                            timesteps.append(raw_data.loc[index, event])
                            trigger = raw_data.loc[index, event+'_Trigger']
                            trigger=str(int(trigger))
                            triggers.append(trigger) # triggers can be used to compare translational and steady-state error
                            labels = np.append(labels,[float(trigger[2])], axis =0)
                            self.prev_label = np.append(self.prev_label,[float(trigger[0])], axis =0)
                            if float(trigger[2]) == 0:
                                print('sitting condition exists!!!!!')
                        index += 1
                    index = 0

                for idx,timestep in enumerate(timesteps):
                    if timestep-window_size-1 >= 0:
                        data = np.array(raw_data.loc[timestep-window_size-1:timestep-2, 'Right_Shank_Ax':'Left_Knee_Velocity'])
                        if processed:
                            img= self.spectrogram2(data)/128.0-1.0
                            self.dataset.append((img,labels[idx]))
                        else:
                            self.dataset.append((data.T,labels[idx]))

                # print(filename, "has been loaded")
        print("load dataset done")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]

        if self.transform:
            img = F.to_pil_image(np.uint8(img))
            img = self.transform(img)
            img = np.array(img)
        return torch.FloatTensor(img), torch.LongTensor(np.array(label) )

    def spectrogram2(self, segmented_data, fs=500,hamming_windowsize=30, overlap = 15):

        vals=[]
        for i in range(0,17):
        	# vals.append([])
	        for x in range(3*i,3*(i+1)):
	            row = segmented_data[:,x]
	            f, t, Sxx = signal.spectrogram(row, fs, window=signal.windows.hamming(hamming_windowsize, True), noverlap=5)
	            tmp, _ = stats.boxcox(Sxx.reshape(-1,1))
	            Sxx = tmp.reshape(Sxx.shape)-np.min(tmp)
	            Sxx = Sxx/np.max(Sxx)*255
	            vals.append(Sxx)

        out = np.stack(vals, axis=0)
        out=out.astype(np.uint8)


        return out


    def spectrogram(self, segmented_data, fs=500, hamming_windowsize=10):
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
            X = 0.2989*X[:,1] + 0.5870*X[:,2] + 0.1140*X[:,3]
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
            X = 0.2989*X[:,1] + 0.5870*X[:,2] + 0.1140*X[:,3]
            vals2.append(X)
            plt.close()

        out1 = np.stack(vals1, axis=2).astype(np.uint8)
        out2 = np.stack(vals2, axis=2).astype(np.uint8)
        out = np.hstack((out1, out2))
        cv2.imshow("ret", out)
        cv2.waitKey(0)
     # print(ret.shape)
        return ret
