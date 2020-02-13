import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset
import torchvision.models as models

from dataset import EnableDataset

import pickle

from sklearn.model_selection import KFold, StratifiedKFold,ShuffleSplit ,train_test_split

import copy
import os
import random


import sys,os
sys.path.append('.')

from utils import *
from networks import *


from itertools import combinations


def run_classifier(mode='bilateral',classifier='CNN',sensor=["imu","emg","goin"],NN_model = None):

	########## SETTINGS  ########################

	BATCH_SIZE = 32
	LEARNING_RATE = 1e-5
	WEIGHT_DECAY = 1e-3
	NUMB_CLASS = 5
	NUB_EPOCH= 200
	numfolds = 10
	DATA_LOAD_BOOL = True

	SAVING_BOOL = False
	MODE_SPECIFIC_BOOL= True

	BAND=10
	HOP=10
	############################################

	print('Number of folds: ', numfolds)


	MODE = mode
	CLASSIFIER = classifier
	SENSOR = sensor
	sensor_str='_'.join(SENSOR)


	MODEL_NAME = './models/Freq-Encoding/bestmodel'+ \
	        		'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'.pth'

	# RESULT_NAME= './results/Freq-Encoding/accuracy'+ \
	        		# '_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'.txt'


	RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER+'_'+MODE+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_accuracy.txt'

	SAVE_NAME= './checkpoints/'+CLASSIFIER+'/'+CLASSIFIER+'_'+MODE+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'mode_secific'+'.pkl'

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

	# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 51),bands=BAND,hop_length=HOP,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE,mode_specific = MODE_SPECIFIC_BOOL)

	BIO_train= EnableDataset(subject_list= ['156'],data_range=(1, 8),bands=BAND,hop_length=HOP,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE,mode_specific = MODE_SPECIFIC_BOOL)


	if SAVING_BOOL:
		save_object(BIO_train,SAVE_NAME)

	# with open(SAVE_NAME, 'rb') as input:
	#     BIO_train = pickle.load(input)

	INPUT_NUM=BIO_train.input_numb

	wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))


	device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
	print('GPU USED?',torch.cuda.is_available())

	if NN_model == 'RESNET18':
		model = MyResNet18() # use resnet
		model.conv1 = nn.Conv2d(INPUT_NUM, 64, kernel_size=5, stride=1, padding=2)
		model.fc = nn.Linear(517 ,NUMB_CLASS)
	else:	
		model = Network_modespecific(INPUT_NUM,NUMB_CLASS)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	num_epoch = NUB_EPOCH

	init_state = copy.deepcopy(model.state_dict())
	init_state_opt = copy.deepcopy(optimizer.state_dict())

	one_hot_embed= torch.eye(5)

	for batch, label, dtype, prevlabels  in tqdm(wholeloader,disable=DATA_LOAD_BOOL):
		X = batch
		y = label
		types = dtype
		prevlabel = prevlabels

	accuracies =[]
	ss_accuracies=[]
	tr_accuracies=[]


	skf = KFold(n_splits = numfolds, shuffle = True)
	i = 0


	train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME)

	for train_index, test_index in skf.split(X, y):

		model.load_state_dict(init_state)
		optimizer.load_state_dict(init_state_opt)

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		types_train, types_test = types[train_index], types[test_index]
		onehot_train, onehot_test = one_hot_embed[prevlabel[train_index]], one_hot_embed[prevlabel[test_index]]


		train_dataset = TensorDataset( X_train, y_train, types_train,onehot_train)
		test_dataset = TensorDataset( X_test, y_test, types_test,onehot_test)

		trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
		testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		print("######################Fold:{}#####################3".format(i+1))
		train_class.train_modesp(trainloader,num_epoch)

		model.load_state_dict(torch.load(MODEL_NAME))

		# print("Evaluate on test set")
		accs,ss_accs,tr_accs=train_class.evaluate_modesp(testloader)
		accuracies.append(accs)
		ss_accuracies.append(ss_accs)
		tr_accuracies.append(tr_accs)

		i +=1

	print('saved on the results')


	# with open(RESULT_NAME, 'w') as f:
	# 	for item in accuracies:
	# 		f.write("%s\n" % item)
	# f.close()


	print('writing...')
	with open(RESULT_NAME, 'w') as f:
		f.write('total ')
		for item in accuracies:
			f.write("%s " % item)
		f.write('\n')
		f.write('steadystate ')
		for item in ss_accuracies:
			f.write("%s " % item)
		f.write('\n')
		f.write('transitional ')
		for item in tr_accuracies:
			f.write("%s " % item)
	f.close()


classifiers=['CNN']
sensors=["imu","emg","goin"]
# modes = ['bilateral','ipsilateral','contralateral']
modes = ['bilateral']
# NNMODEL = 'RESNET18'
NNMODEL = 'bionet'
for classifier in classifiers:
	for i in range(3,4):
		for combo in combinations(sensors,i):
			sensor = [item for item in combo]
			for mode in modes:
				print(classifier, sensor, mode)
				run_classifier(mode=mode,classifier=classifier,sensor=sensor,NN_model = NNMODEL)