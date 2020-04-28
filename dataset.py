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
    transform: optional transform to apply to the data
    '''
    def __init__(self, dataDir='./Data/', subject_list=['156'], model_type="CNN",mode_specific=False, data_range=(1, 51), window_size=500,  sensors=["imu","emg", "goin"], mode="bilateral", transform=None,bands=None,hop_length=None,phaselabel=None,prevlabel=None,delay=0,time_series=False):
        self.model_type = model_type
        if self.model_type == "CNN" or "LSTM" or "DeepConvLSTM" or "LIRLSTM": 
            print("    range: [%d, %d)" % (data_range[0], data_range[1]))
            self.dataset = []
            self.prev_label = np.array([], dtype=np.int64)
            self.img_data_stack=np.empty((51, 3, 4, 51), dtype=np.int64)
            self.transform = transform
            self.input_numb = 0
            self.mode_specific=mode_specific

            exclude_list = [
                # # 'AB194_Circuit_009',
                # # 'AB194_Circuit_017',
                # # 'AB194_Circuit_018',
                # # 'AB194_Circuit_026',
                # # "AB194_Circuit_033",
                # # "AB194_Circuit_038",
                # # "AB193_Circuit_022",
                # # "AB193_Circuit_043",
                # # "AB192_Circuit_034",
                # 'AB190_Circuit_013',
                # # 'AB190_Circuit_014',
                # # 'AB190_Circuit_037',
                # 'AB190_Circuit_045',
                # # "AB191_Circuit_001",
                # 'AB191_Circuit_002',
                # 'AB191_Circuit_022',
                # # "AB191_Circuit_047",
                # # "AB191_Circuit_049",
                # "AB189_Circuit_004",
                # # "AB189_Circuit_024",
                # "AB189_Circuit_032",
                # # "AB189_Circuit_035",
                # # "AB188_Circuit_027",
                # "AB188_Circuit_032",
                # "AB186_Circuit_002",
                # # "AB186_Circuit_004",
                # # "AB186_Circuit_016",
                # # "AB186_Circuit_050",
                # # "AB185_Circuit_002",
                # "AB185_Circuit_008",
                # "AB185_Circuit_010",
                # "AB156_Circuit_005",
                # "AB156_Circuit_050"
            ]

            for subjects in subject_list:
                for i in range(data_range[0], data_range[1]):
                    filename = dataDir +'AB' + subjects+'/Processed/'+'AB' + subjects+ '_Circuit_%03d_post.csv'% i
                    if not os.path.exists(filename) or ('AB' + subjects+ '_Circuit_%03d_post'% i) in exclude_list:
                        print(filename, 'not found or excluded')
                        continue
                    raw_data = pd.read_csv(filename)
                    # pdb.set_trace()
                    # segmented_data = np.array([], dtype=np.int64).reshape(0,window_size,48)
                    labels = np.array([], dtype=np.int64)
                    timestep_type = []
                    timesteps = []
                    triggers = []
                    index = 0
                    gait_event_types = []

                    gait_events = ['Right_Heel_Contact','Right_Toe_Off','Left_Heel_Contact','Left_Toe_Off']
                    for event in gait_events:
                        while not pd.isnull(raw_data.loc[index, event]):
                            trigger = raw_data.loc[index, event+'_Trigger']
                            trigger=str(int(trigger))
                            if float(trigger[2]) != 6 and float(trigger[0]) !=6:
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

                            # regex = "(?=Mode"
                            regex = "(?=!Mode"
                            if "imu" in sensors:
                                regex += "|.*A[xyz].*"
                            if "goin" in sensors:
                                regex += "|.*G[xyz].*|.*Ankle.*|.*Knee.*"
                            if "emg" in sensors:
                                regex += "|.*TA.*|.*MG.*|.*SOL.*|.*BF.*|.*ST.*|.*VL.*|.*RF.*"
                            regex += ")"
                            data = data.filter(regex=regex, axis=1)

                            data = np.array(data)
                            self.input_numb=np.shape(data)[1]
                            
                            if time_series:
                                data = (data-np.mean(data, axis=0))/np.std(data, axis=0)
                                if self.mode_specific:
                                    self.dataset.append((data.T,labels[idx],timestep_type[idx],int(self.prev_label[idx])))
                                else:
                                    self.dataset.append((data.T,labels[idx], timestep_type[idx]))

                            else:
                                img= self.melspectrogram(data,bands=bands ,hop_length=hop_length)
                                if self.mode_specific:
                                    self.dataset.append((img,labels[idx],timestep_type[idx],int(self.prev_label[idx])))
                                else:
                                    self.dataset.append((img,labels[idx], timestep_type[idx]))
        else:
            self.dataset = []
            self.prev_label = np.array([], dtype=np.int64)

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


                    # while not pd.isnull(raw_data.loc[index,'Trigger']):
                    # pdb.set_trace()
                    for index in range(0,raw_data.shape[0]):
                        trigger = raw_data.loc[index,'Trigger']
                        trigger=str(int(trigger))
                        phase = raw_data.loc[index,'Leg Phase']
                        if trigger[0] == trigger[2]:
                            timestep_type.append(1)
                        else:
                            timestep_type.append(0)

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

                                # regex = "(?=Mode|.*Ankle.*|.*Knee.*"
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

                                # regex = "(?=Mode|.*Ankle.*|.*Knee.*"
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

                                # if float(trigger[0]) == 4:
                                #     print('***********',trigger[0])
                                self.prev_label = np.append(self.prev_label,[float(trigger[0])], axis =0)

                                self.dataset.append((data.T,label, timestep_type[-1]))



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.model_type == "CNN" or "LIRLSTM":
            if self.mode_specific:
                img, label, timestep_type, prev__label= self.dataset[index]
                if self.transform:
                    img = F.to_pil_image(np.uint8(img))
                    img = self.transform(img)
                    img = np.array(img)
                    # pdb.set_trace()
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

    def spectrogram2(self, segmented_data, fs=500,hamming_windowsize=30, overlap = 15):
        vals = []
        for i in range(0,17):
	        for x in range(3*i,3*(i+1)):
	            row = segmented_data[:,x]
	            f, t, Sxx = signal.spectrogram(row, fs, window=signal.windows.hamming(hamming_windowsize, True), noverlap=5)
	            tmp, _ = stats.boxcox(Sxx.reshape(-1,1))
	            Sxx = tmp.reshape(Sxx.shape)-np.min(tmp)
	            Sxx = Sxx/np.max(Sxx)*255
	            vals.append(Sxx)
        out = np.stack(vals, axis=0)
        return out

    def melspectrogram(self, segmented_data, fs=500,bands=64 ,hop_length=50):

        ###### STACKING UP MULTIPLE SPECTOGRAM APPROACH!

        vals = []
        # for i in range(0,17):
        #     for x in range(3*i,3*(i+1)):
        for x in range(0,np.shape(segmented_data)[1]):

                row = segmented_data[:,x]
                melspec_full = librosa.feature.melspectrogram(y=row,sr=fs,n_fft=hop_length*2, hop_length=hop_length,n_mels=bands)
                logspec_full = librosa.amplitude_to_db(melspec_full)
                # logspec_delta = librosa.feature.delta(logspec_full) # add derivative

                ## plotting spectro and melspectro
#                 if x == 30:
#                     plt.figure(figsize=(10,8))
#                     plt.rcParams['font.family'] = 'Times New Roman'  
#                     plt.rcParams.update({'font.size': 31})
#                     # D = librosa.amplitude_to_db(np.abs(librosa.stft(row)), ref=np.max)
#                     # librosa.display.specshow(D, x_axis='s',y_axis='mel',sr=fs,fmax=fs/2,cmap='viridis')
#                     f, t, Sxx=signal.spectrogram(row, fs, window=signal.windows.hamming(hop_length*2, True),nfft=hop_length*2, noverlap=hop_length)
#                     # plt.imshow(spec,aspect='auto',origin='lower',extent=[times.min(),times.max(),freqs.min(),freqs.max()])
#                     plt.pcolormesh(t, f, 10*np.log10(Sxx),vmin=-80, vmax=0)
#                     # plt.pcolormesh(t, f, Sxx,norm = matplotlib.colors.Normalize(0,1))
#                     plt.colorbar(format='%+2.0f dB')
#                     plt.xlabel('Time (s)')
#                     plt.ylabel('Hz')
#                     # plt.title('Linear-frequency power spectrogram')
#                     plt.yticks(np.array([0,50,100,150,200]), ['0','50','100','150','200'])
#                     # plt.savefig('./spectro.png')
#                     plt.show()

#                     plt.figure(figsize=(10,4))
#                     S_dB = librosa.amplitude_to_db(melspec_full, ref=np.max)
#                     librosa.display.specshow(S_dB,x_axis='s',hop_length=10,y_axis='linear',sr=fs,fmax=fs/2,cmap='viridis')
#                     plt.colorbar(format='%+2.0f dB')

#                     locs, labels = plt.xticks()  
#                     plt.yticks(np.array([0,50,100,150,200]), ['0','50','100','150','200'])
#                     plt.xticks(np.array([0.25,0.5,0.75]), ['0.25','0.50','0.75'])
#                     plt.show()
#                     pdb.set_trace()


                vals.append(logspec_full)
        return vals


    def cwt(self, segmented_data, fs=500,hamming_windowsize=30, overlap = 15):
        vals = []
        for i in range(0,17):
            for x in range(3*i,3*(i+1)):
                row = segmented_data[:,x]
                widths = np.arange(1,101)
                cwtmatr = signal.cwt(row, signal.ricker, widths)
                print(cwtmatr.shape, np.min(cwtmatr), np.max(cwtmatr))
                cwtmatr = cwtmatr-np.min(cwtmatr)
                cwtmatr = cwtmatr/np.max(cwtmatr)*255
                vals.append(cwtmatr)


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
        return ret
