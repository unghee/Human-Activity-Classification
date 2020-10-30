import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset



import pickle

from sklearn.model_selection import KFold, StratifiedKFold,ShuffleSplit ,train_test_split
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
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Used to add a hook to our model. The hook is a function that will run
# during our model execution.
class SaveFeatures():
	features=None
	def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

	def hook_fn(self, module, input, output):
		self.features = ((output.cpu()).data).numpy()

	def remove(self):
		self.hook.remove()
	# Save the first channel the activation map
	def plot_activation(self, filename):
		img = Image.fromarray(self.features[0,1], 'L')
		plt.imshow(img)
		plt.show()
		img.save(filename + '.png')



def run_classifier(mode='bilateral',classifier='CNN',sensor=["imu","emg","goin"]):

	########## SETTINGS  ########################

	BATCH_SIZE = 32
	LEARNING_RATE = 1e-5
	WEIGHT_DECAY = 1e-3
	NUMB_CLASS = 5
	NUB_EPOCH= 200
	numfolds = 10
	DATA_LOAD_BOOL = True

	SAVING_BOOL = True
	############################################



	MODE = mode
	CLASSIFIER = classifier
	SENSOR = sensor
	sensor_str='_'.join(SENSOR)


	MODEL_NAME = './models/Freq-Encoding/bestmodel'+ \
	        		'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'.pth'

	# RESULT_NAME= './results/Freq-Encoding/accuracy'+ \
	        		# '_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'.txt'


	RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER+'_'+MODE+'_'+sensor_str+'_accuracy.txt'

	SAVE_NAME= './checkpoints/'+CLASSIFIER+'/'+CLASSIFIER+'_'+MODE+'_'+sensor_str+'.pkl'

	if not os.path.exists('./models/Freq-Encoding'):
		os.makedirs('./models/Freq-Encoding')


	if not os.path.exists('./results/'+CLASSIFIER):
		os.makedirs('./results/'+CLASSIFIER)

	if not os.path.exists('./checkpoints/'+CLASSIFIER):
		os.makedirs('./checkpoints/'+CLASSIFIER)

	# if not os.path.exists('./results/Freq-Encoding'):
	# 	os.makedirs('./results/Freq-Encoding')


	# Load the dataset and train, val, test splits
	print("Loading datasets...")

	# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=10,hop_length=10,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE)
	BIO_train= EnableDataset(subject_list= ['156'],data_range=(1, 2),bands=10,hop_length=10,model_type='CNN')

	INPUT_NUM=BIO_train.in_channels
	
	# with open('BIO_train_melspectro_500s_bands_16_hop_length_27.pkl', 'rb') as input:
	#     BIO_train = pickle.load(input)

	if SAVING_BOOL:
		save_object(BIO_train,SAVE_NAME)


	wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))


	device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
	print('GPU USED?',torch.cuda.is_available())
	model = Network(INPUT_NUM,NUMB_CLASS)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	num_epoch = NUB_EPOCH

	init_state = copy.deepcopy(model.state_dict())
	init_state_opt = copy.deepcopy(optimizer.state_dict())


	for batch, label, dtype in tqdm(wholeloader,disable=DATA_LOAD_BOOL):
		X = batch
		y = label
		types = dtype

	accuracies =[]
	class_accs = [0] * NUMB_CLASS



	train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME)


	print('saved on the results')
	print("average:")
	for i in range(len(class_accs)):
		if class_accs[i] == 0:
			print("Class {} has no samples".format(i))
		else:
			print("Class {} accuracy: {}".format(i, class_accs[i]/numfolds))


	model.load_state_dict(torch.load('./models/Freq-Encoding/bestmodel_BATCH_SIZE32_LR1e-05_WD0.001_EPOCH200_BAND10_HOP10.pth', map_location='cpu'))

	def normalize_output(img):
		if img.min()==0 and img.max() ==0:
			pass
		else:
		    img = img - img.min()
		    img = img / img.max()
		return img

	# Save one channel from the first datum in the dataset

	# fig1, axs = plt.subplots(3,figsize=(6,20))
	fig1, axs = plt.subplots(3)
	fontname = 	 'Times New Roman' 
	plt.rcParams['font.family'] = fontname
	# representative image of IMU	Right_Shank_Ax
	im=axs[0].imshow(normalize_output(BIO_train[0][0][0]))
	axs[0].invert_yaxis()
	axs[0].spines['top'].set_visible(False)
	axs[0].spines['right'].set_visible(False)
	axs[0].spines['bottom'].set_visible(False)
	axs[0].spines['left'].set_visible(False)

	axs[0].get_xaxis().set_visible(False)
	# cb=fig1.colorbar(im, ax=axs[0])
	# cb.outline.set_visible(False)
	title=axs[0].set_title('IMU',fontname=fontname) 
	# title.rcParams['font.family'] = fontname
	# EMG Right_TA
	im2=axs[1].imshow(normalize_output(BIO_train[0][0][30]))
	axs[1].invert_yaxis()
	axs[1].spines['top'].set_visible(False)
	axs[1].spines['right'].set_visible(False)
	axs[1].spines['bottom'].set_visible(False)
	axs[1].spines['left'].set_visible(False)	
	axs[1].get_xaxis().set_visible(False)
	axs[1].set_title('EMG',fontname=fontname) 
	# GION Right_TA
	im3=axs[2].imshow(normalize_output(BIO_train[0][0][44]))
	axs[2].invert_yaxis()
	axs[2].spines['top'].set_visible(False)
	axs[2].spines['right'].set_visible(False)
	axs[2].spines['bottom'].set_visible(False)
	axs[2].spines['left'].set_visible(False)
	axs[2].set_title('Goniometer',fontname=fontname)
	locs, labels = plt.xticks()  
	# plt.xticks(np.array([0,25,50]), ['0','0.5','1'])
	# plt.xlabel('Time (s)')

	locs, labels = plt.yticks()  
	# plt.yticks(np.array([0,25,50]), ['0','0.5','1'])

	# plt.xlabel('Number of Pixels')
	# axs[0].set_ylabel('Number of Pixels')

	# axs[0].set_yticks(np.array([0,2,4,6,8]))



	def pix_to_hz(x):
		y=x*25
		return y
	def hz_to_pix(x):
		y=x/25
		return y




	ax2 = axs[0].secondary_yaxis('right', functions=(pix_to_hz,hz_to_pix))
	# ax2 = axs[0].twinx()
	ax2.set_yticks(np.array([0,50/25,100/25,150/25,200/25]))
	ax2.set_yticklabels(['0','50','100','150','200'])
	# ax2.yaxis.set_ticks(np.array([0,50/25,100/25,150/25,200/25]))
	# ax2.yaxis.set_tickslabels(['0','50','100','150','200'])
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.spines['bottom'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	ax2.set_ylabel('Hz')


	# plt.yticks(np.array([0,50/25,100/25,150/25,200/25,250/25]), ['0','50','100','150','200','250'])
	fig1.text(0.5, 0.04, 'Number of Time Frame', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
	fig1.text(0.04, 0.5, 'Number of Mel-bins', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
	


# Visualize feature maps
	activation = {}
	def get_activation(name):
		def hook(model, input, output):
			activation[name] = output.detach()
		return hook	


	model.sclayer1.register_forward_hook(get_activation('sclayer1'))

	data = BIO_train[0][0]
	data.unsqueeze_(0)
	output = model(data)

	act = activation['sclayer1'].squeeze()

	fig, axarr = plt.subplots(5,2)
	# idxes = random.sample(range(0, 128), 10)
	idxes = np.array([4,63,32,5,56,8,119,105,110,48])

	col =0
	for idx, idxe in enumerate(idxes):

	    rem=idx%5 
	    im5=axarr[rem,col].imshow(normalize_output(act[idxe]),interpolation='bilinear', cmap='jet')
	    axarr[rem,col].invert_yaxis()
	    axarr[rem,col].spines['top'].set_visible(False)
	    axarr[rem,col].spines['right'].set_visible(False)
	    axarr[rem,col].spines['bottom'].set_visible(False)
	    axarr[rem,col].spines['left'].set_visible(False)
	    # axarr[rem,col].get_xaxis().set_visible(False)
	    # axarr[rem,col].get_yaxis().set_visible(False)
	    if not (idx % 5 ==4):
	    	axarr[rem,col].get_xaxis().set_visible(False)
	    if idx >4:
	    	axarr[rem,col].get_yaxis().set_visible(False)
	    if idx %  5==4:
	    	col +=1
	    print(idx,idxe)

	fontname = 	 'Times New Roman'   	
	for ax in axarr.flatten():
	    labels = ax.get_xticklabels() + ax.get_yticklabels()
	    [label.set_fontname(fontname) for label in labels]

	for ax in axs.flatten():
	    labels = ax.get_xticklabels() + ax.get_yticklabels()
	    [label.set_fontname(fontname) for label in labels]


	# cbar_ax = fig1.add_axes([0.9, 0.15, 0.05, 0.7])
	# cb=fig1.colorbar(im5, cax=cbar_ax)


	fig.text(0.5, 0.04, 'Number of Pixels', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
	fig.text(0.04, 0.5, 'Number of Pixels', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])

	fig1.savefig('./input.png')
	fig.savefig('./activation.png')
	plt.show()



	with open(RESULT_NAME, 'w') as f:
		for item in accuracies:
			f.write("%s\n" % item)
	f.close()

# Code for the different subgroups
sensors=["imu","emg","goin"]

run_classifier(mode='bilateral',classifier='CNN',sensor=sensors)