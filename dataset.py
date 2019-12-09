import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from scipy import signal, stats
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np




class EnableDataset(Dataset):
    def __init__(self, dataDir='./Data/' ,subject_list=['156'], data_range=(1, 10), window_size=500, stride=500, delay=500, processed=False):

        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.dataset = []
        # self.img_data_stack=np.array([],shape=(51, 3, 4, 51), dtype=np.int64)
        self.img_data_stack=np.empty((51, 3, 4, 51), dtype=np.int64)
        # subject_list= ['156','185']
        self.dataset = []
        for subjects, ranges in zip(subject_list, data_range):
            print(subjects)
            for i in range(1, ranges):   
                raw_data = pd.read_csv(dataDir +'AB' + subjects+'/Processed/'+'AB' + subjects+ '_Circuit_%03d_post.csv'% i)


                segmented_data = np.array([], dtype=np.int64).reshape(0,window_size,48)
                labels = np.array([], dtype=np.int64)
                timesteps = []
                triggers = []
                index = 0

                gait_events = ['Right_Heel_Contact','Right_Toe_Off','Left_Heel_Contact','Left_Toe_Off']
                for event in gait_events:
                	while not pd.isnull(raw_data.loc[index, event]):
	                    timesteps.append(raw_data.loc[index, event])
	                    trigger = raw_data.loc[index, event+'_Trigger']
	                    trigger=str(int(trigger))
	                    triggers.append(trigger) # triggers can be used to compare translational and steady-state error
	                    labels = np.append(labels,[float(trigger[2])], axis =0)
	                    index += 1
	                index = 0

                # while not pd.isnull(raw_data.loc[index, 'Right_Heel_Contact']):
                #     timesteps.append(raw_data.loc[index, 'Right_Heel_Contact'])
                #     trigger = raw_data.loc[index, 'Right_Heel_Contact_Trigger']
                #     trigger=str(int(trigger))
                #     triggers.append(trigger) # triggers can be used to compare translational and steady-state error
                #     labels = np.append(labels,[float(trigger[2])], axis =0)
                #     index += 1
                # index = 0
                # while not pd.isnull(raw_data.loc[index, 'Right_Toe_Off']):
                #     timesteps.append(raw_data.loc[index, 'Right_Toe_Off'])
                #     trigger = raw_data.loc[index, 'Right_Toe_Off_Trigger']
                #     trigger=str(int(trigger))
                #     triggers.append(trigger) # triggers can be used to compare translational and steady-state error
                #     labels = np.append(labels,[float(trigger[2])], axis =0)
                #     index += 1
                # index = 0
                # while not pd.isnull(raw_data.loc[index, 'Left_Heel_Contact']):
                #     timesteps.append(raw_data.loc[index, 'Left_Heel_Contact'])
                #     trigger = raw_data.loc[index, 'Left_Heel_Contact_Trigger']
                #     trigger=str(int(trigger))
                #     triggers.append(trigger) # triggers can be used to compare translational and steady-state error
                #     labels = np.append(labels,[float(trigger[2])], axis =0)
                #     index += 1
                # index = 0 
                # while not pd.isnull(raw_data.loc[index, 'Left_Toe_Off']):
                #     timesteps.append(raw_data.loc[index, 'Left_Toe_Off'])
                #     trigger = raw_data.loc[index, 'Left_Toe_Off_Trigger']
                #     trigger=str(int(trigger))
                #     triggers.append(trigger) # triggers can be used to compare translational and steady-state error
                #     labels = np.append(labels,[float(trigger[2])], axis =0)
                #     index += 1
                # index = 0 

                for idx,timestep in enumerate(timesteps):
                    if timestep-window_size-1 >= 0:
                        # labels = np.append(labels, [raw_data.loc[timestep, 'Mode']], axis=0)
                        data = np.array(raw_data.loc[timestep-window_size-1:timestep-2, 'Right_Shank_Ax':'Left_Knee_Velocity'])
                        img= self.spectrogram2(data)/128.0-1.0
                        #img = cv2.resize(img.transpose(1,2,0), None, fx=2, fy=2).transpose(2,0,1)
                        #print(img.shape)
                        #cv2.imshow("img", img.transpose(1,2,0))
                        #cv2.waitKey(0)
                        # f, t, Sxx = signal.spectrogram(data, 500)
                        # Pxx, freqs, bins, im = plt.specgram(data, Fs=500)
                        # cv2.imshow("test", Pxx)
                        # cv2.waitKey(0)
                        #img=np.asarray(img).transpose(2, 1, 0)/128.0-1.0

                        self.dataset.append((img,labels[idx]))
        print("load dataset done")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return torch.FloatTensor(img), torch.LongTensor(np.array(label) )

    def spectrogram2(self, segmented_data, fs=500,hamming_windowsize=30, overlap = 15):
        #vals=[[],[],[],[],[],[]]
        vals = []
        for i in range(0,17):
	        for x in range(3*i,3*(i+1)):
	            row = segmented_data[:,x]
	            f, t, Sxx = signal.spectrogram(row, fs, window=signal.windows.hamming(hamming_windowsize, True), noverlap=5)
	            tmp, _ = stats.boxcox(Sxx.reshape(-1,1))
	            Sxx = tmp.reshape(Sxx.shape)-np.min(tmp)
	            Sxx = Sxx/np.max(Sxx)*255
	            vals.append(Sxx)
        #print(vals[0][0].shape)

        #layer1 = np.hstack((np.vstack((vals[0][0],vals[1][0],vals[2][0])), np.vstack((vals[0][0],vals[1][0],vals[2][0]))))
        #layer2 = np.hstack((np.vstack((vals[0][1],vals[1][1],vals[2][1])), np.vstack((vals[0][1],vals[1][1],vals[2][1]))))
        #layer3 = np.hstack((np.vstack((vals[0][2],vals[1][2],vals[2][2])), np.vstack((vals[0][2],vals[1][2],vals[2][2]))))
        #out = np.stack((layer1, layer2, layer3), axis=0)
        out = np.stack(vals, axis=0)
        #print(out.shape)
        out=out.astype(np.uint8)
        #print(out.shape)
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
