import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset, ConcatDataset



import pickle

from sklearn.model_selection import KFold, StratifiedKFold,ShuffleSplit ,train_test_split
from sklearn.metrics import confusion_matrix, classification_report

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



def run_classifier(mode='bilateral',classifier='CNN',sensor=["imu","emg","goin"],NN_model=None):

	########## SETTINGS  ########################

	BATCH_SIZE = 32
	LEARNING_RATE = 1e-5
	WEIGHT_DECAY = 1e-3
	NUMB_CLASS = 5
	NUB_EPOCH= 200
	numfolds = 10
	DATA_LOAD_BOOL = True
	BAND=10
	HOP=10
	# BAND=16,HOP=27
	SAVING_BOOL = True
	############################################

	MODE = mode
	CLASSIFIER = classifier
	SENSOR = sensor
	sensor_str='_'.join(SENSOR)

	MODEL_NAME = './models/Freq-Encoding/bestmodel'+ \
	        		'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_subjects.pth'


	SAVE_NAME= './checkpoints/'+CLASSIFIER+'/'+CLASSIFIER +'_'+MODE+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'subjects.pkl'
	RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER + NN_model+'_'+MODE+'_'+sensor_str+'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_subjects_accuracy.txt'
	# Load the dataset and train, val, test splits
	print("Loading datasets...")

	subjects = ['156','185','186','188','189','190', '191', '192', '193', '194']


	if SAVING_BOOL:
		subject_data = []
		for subject in subjects:
			subject_data.append(EnableDataset(subject_list= [subject],data_range=(1, 51),bands=BAND,hop_length=HOP,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE))

		save_object(subject_data,SAVE_NAME)
	else:
		with open(SAVE_NAME, 'rb') as input:
		    subject_data = pickle.load(input)


	INPUT_NUM=subject_data[0].input_numb

	device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
	print('GPU USED?',torch.cuda.is_available())

	if NN_model == 'RESNET18':
		model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True) # use resnet
		num_ftrs = model.fc.in_features
		# model.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
		top_layer= nn.Conv2d(INPUT_NUM, 3, kernel_size=5, stride=1, padding=2)
		model = nn.Sequential(top_layer,model)
		model.fc = nn.Linear(num_ftrs, NUMB_CLASS)

	else:
		model = Network(INPUT_NUM,NUMB_CLASS)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	num_epoch = NUB_EPOCH

	init_state = copy.deepcopy(model.state_dict())
	init_state_opt = copy.deepcopy(optimizer.state_dict())

	accuracies =[]

	ss_accuracies=[]
	tr_accuracies=[]

	class_accs = [0] * NUMB_CLASS

	subject_numb = []

	# skf = KFold(n_splits = numfolds, shuffle = True)
	skf = KFold(n_splits = len(subject_data), shuffle = True)
	i = 0

	train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME)

	tests=[]
	preds=[]
	# for train_index, test_index in skf.split(X, y, types):
	for train_index, test_index in skf.split(subject_data):
		print(train_index,test_index)

		print(train_index,test_index)

		model.load_state_dict(init_state)
		optimizer.load_state_dict(init_state_opt)

		train_set = [subject_data[i] for i in train_index]
		test_set = [subject_data[i] for i in test_index]
		BIO_train = torch.utils.data.ConcatDataset(train_set)
		wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))

		for batch, label, dtype in tqdm(wholeloader,disable=DATA_LOAD_BOOL):
			X_train = batch
			y_train = label
			types_train = dtype
		BIO_train = None
		train_set = None

		BIO_test = torch.utils.data.ConcatDataset(test_set)
		wholeloader = DataLoader(BIO_test, batch_size=len(BIO_test))

		for batch, label, dtype in tqdm(wholeloader,disable=DATA_LOAD_BOOL):
			X_test = batch
			y_test = label
			types_test = dtype
		BIO_test = None
		test_set = None

		train_dataset = TensorDataset( X_train, y_train, types_train)
		test_dataset = TensorDataset( X_test, y_test, types_test)

		trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
		testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		print("######################Fold:{}#####################".format(i+1))

		train_class.train(trainloader,num_epoch)

		model.load_state_dict(torch.load(MODEL_NAME))

		accs,ss_accs,tr_accs,pred,test,class_acc=train_class.evaluate(testloader)
		accuracies.append(accs)
		ss_accuracies.append(ss_accs)
		tr_accuracies.append(tr_accs)

		preds.extend(pred)
		tests.extend(test)

		subject_numb.append(test_index[0])


		for j in range(len(class_accs)):
			class_accs[j] += class_acc[j]

		del  test_dataset, train_dataset, trainloader, testloader


		i +=1

	# print("average:")
	# for i in range(len(class_accs)):
	# 	if class_accs[i] == 0:
	# 		print("Class {} has no samples".format(i))
	# 	else:
	# 		print("Class {} accuracy: {}".format(i, class_accs[i]/numfolds))

	print("Accuracies")
	for item in accuracies:
		print(item)

	print("Steady state")
	for item in ss_accuracies:
		print(item)

	print("Translational")
	for item in tr_accuracies:
		print(item)


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
		f.write('\n')
		f.write('subject_numb ')
		for item in subject_numb:
			f.write("%s " % item)

	f.close()



	conf= confusion_matrix(tests, preds)
	print(conf)
	print(classification_report(tests, preds, digits=3))

	return conf



classifiers=['CNN']
sensors=["imu","emg","goin"]
sensor_str='_'.join(sensors)
modes = ['bilateral']
NN = 'bionet'
for classifier in classifiers:
	for i in range(3,4):
		for combo in combinations(sensors,i):
			sensor = [item for item in combo]
			for mode in modes:
				print(classifier, sensor, mode)
				confusion=run_classifier(mode=mode,classifier=classifier,sensor=sensor,NN_model=NN)


