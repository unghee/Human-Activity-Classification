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

import copy
import os
import random


import sys,os
sys.path.append('.')

from dataset import EnableDataset
from utils import *
from networks import *

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
NUMB_CLASS = 5
NUB_EPOCH= 200
numfolds = 10
DATA_LOAD_BOOL = True

SAVING_BOOL = True

MODE = 'bilateral'
CLASSIFIER = 'Random'
SENSOR = ["imu","emg","goin"]
sensor_str='_'.join(SENSOR)

RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER + '_subjects_accuracy.txt'

if not os.path.exists('./results/'+CLASSIFIER):
		os.makedirs('./results/'+CLASSIFIER)

# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 10),bands=10,hop_length=10,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE)
# BIO_train= EnableDataset(subject_list= ['156'],data_range=(1, 4),bands=10,hop_length=10,mode_specific = True,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE)

subjects = ['156','185','186','188','189','190', '191', '192', '193', '194']

if SAVING_BOOL:
	subject_data = []
	for subject in subjects:
		subject_data.append(EnableDataset(subject_list= [subject],model_type=CLASSIFIER,sensors=SENSOR,mode=MODE))

	save_object(subject_data,'./checkpoints/count_Data_features.pkl')	
else:
	with open('./checkpoints/count_Data_features.pkl', 'rb') as input:
		   subject_data = pickle.load(input)




skf = KFold(n_splits = numfolds, shuffle = True)
i = 0

overall_accs = []
ss_accs = []
tr_accs = []

for train_index, test_index in skf.split(subject_data):

	print(train_index, test_index)

	train_vals = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
	test_vals = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

	print("######################Fold:{}#####################".format(i+1))

	train_set = [subject_data[i] for i in train_index]
	test_set = [subject_data[i] for i in test_index]
	BIO_train = torch.utils.data.ConcatDataset(train_set)
	wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))
	for batch, label, trigger,dtype in tqdm(wholeloader):
		X_train = batch
		y_train = label
		types_train = dtype
		trigger_train = trigger

	BIO_test = torch.utils.data.ConcatDataset(test_set)
	wholeloader = DataLoader(BIO_test, batch_size=len(BIO_train))
	for batch, label, trigger,dtype in tqdm(wholeloader):
		X_test = batch
		y_test = label
		types_test = dtype
		trigger_test = trigger


	train_dataset = TensorDataset( X_train, y_train, trigger_train)
	test_dataset = TensorDataset( X_test, y_test, trigger_test)

	for img, labels, trigger in train_dataset:
		train_vals[int(trigger)-1][int(labels)-1]+=1

	for img, labels, trigger in test_dataset:
		test_vals[int(trigger)-1][int(labels)-1]+=1

	print(test_vals)
	test_vals=np.array(test_vals)
	train_vals=np.array(train_vals)

	if CLASSIFIER =='Random_modespecific':

		if np.argmax(train_vals,1).all() == np.array([0,1,2,3,4]).all():

			overall_acc= np.sum(np.max(test_vals,0))/np.sum(test_vals)
			# pdb.set_trace()
			overall_accs.append(overall_acc)
			print(overall_acc)

			if np.max(train_vals).all() == np.diag(train_vals).all():
				ss_acc = 1
				tr_acc = 0 
				ss_accs.append(ss_acc)
				tr_accs.append(tr_acc)

		else:
			overall_acc = Nan
			overall_accs.append(overall_acc)


	elif CLASSIFIER =="Random":

		if np.argmax(train_vals) == 0: 
			overall_acc= np.sum(test_vals[:,0])/np.sum(test_vals)
			overall_accs.append(overall_acc)

			ss_acc = test_vals[0][0]/np.sum(np.diag(test_vals))
			tr_acc = np.sum(test_vals[1:,0])/(np.sum(test_vals)-np.sum(np.diag(test_vals)))

			ss_accs.append(ss_acc)
			tr_accs.append(tr_acc)
		else: 
			overall_acc = Nan
			overall_accs.append(overall_acc)
		


		print('overall.{}, ss.{}, tr,{}'.format(overall_acc,ss_acc,tr_acc))

	i +=1


print('writing...')
with open(RESULT_NAME, 'w') as f:
	f.write('total ')
	for item in overall_accs:
		f.write("%s " % item)
	f.write('\n')
	f.write('steadystate ')
	for item in ss_accs:
		f.write("%s " % item)
	f.write('\n')
	f.write('transitional ')
	for item in tr_accs:
		f.write("%s " % item)
f.close()

