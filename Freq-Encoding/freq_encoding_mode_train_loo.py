import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset,  ConcatDataset
import torchvision.models as models

# from dataset import EnableDataset

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

	SAVING_BOOL = True
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


	# RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER+'_'+MODE+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_subjects_accuracy.txt'
	RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER + NN_model+'_'+MODE+'_'+sensor_str+'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_mode_specific_subjects_accuracy.txt'

	SAVE_NAME= './checkpoints/'+CLASSIFIER+'/'+CLASSIFIER+'_'+MODE+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'mode_secific'+'subjects.pkl'

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

	subjects = ['156','185','186','188','189','190', '191', '192', '193', '194']

	spectrogramTime = 0.0
	if SAVING_BOOL:
		subject_data = []
		for subject in subjects:
			subject_data.append(EnableDataset(subject_list= [subject],data_range=(1, 51),bands=BAND,hop_length=HOP,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE,mode_specific = MODE_SPECIFIC_BOOL))
			spectrogramTime += subject_data[-1].spectrogramTime
		save_object(subject_data,SAVE_NAME)
	else:
		with open(SAVE_NAME, 'rb') as input:
			subject_data = pickle.load(input)
	spectrogramTime = spectrogramTime / len(subjects)

	INPUT_NUM=subject_data[0].input_numb

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

	accuracies =[]
	ss_accuracies=[]
	tr_accuracies=[]
	subject_numb = []


	skf = KFold(n_splits = numfolds, shuffle = True)
	i = 0


	train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME)
	inferenceTime = 0.0

	for train_index, test_index in skf.split(subject_data):
		print(train_index,test_index)

		model.load_state_dict(init_state)
		optimizer.load_state_dict(init_state_opt)

		train_set = [subject_data[i] for i in train_index]
		test_set = [subject_data[i] for i in test_index]
		BIO_train = torch.utils.data.ConcatDataset(train_set)
		wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))
		for batch, label, dtype, prevlabel in tqdm(wholeloader,disable=DATA_LOAD_BOOL):
			X_train = batch
			y_train = label
			types_train = dtype
			prevlabel_train = prevlabel
		BIO_train = None
		train_set = None

		BIO_test = torch.utils.data.ConcatDataset(test_set)
		wholeloader = DataLoader(BIO_test, batch_size=len(BIO_test))
		for batch, label, dtype, prevlabel in tqdm(wholeloader,disable=DATA_LOAD_BOOL):
			X_test = batch
			y_test = label
			types_test = dtype
			prevlabel_test = prevlabel
		BIO_test = None
		test_set = None

		train_dataset = TensorDataset( X_train, y_train, types_train)
		test_dataset = TensorDataset( X_test, y_test, types_test)

		trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
		testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		model.load_state_dict(init_state)
		optimizer.load_state_dict(init_state_opt)

		onehot_train, onehot_test = one_hot_embed[prevlabel_train], one_hot_embed[prevlabel_test]


		train_dataset = TensorDataset( X_train, y_train, types_train,onehot_train)
		test_dataset = TensorDataset( X_test, y_test, types_test,onehot_test)

		trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
		testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		print("######################Fold:{}#####################3".format(i+1))
		train_class.train_modesp(trainloader,num_epoch)

		model.load_state_dict(torch.load(MODEL_NAME))

		# print("Evaluate on test set")
		accs,ss_accs,tr_accs,inf_time=train_class.evaluate_modesp(testloader)
		accuracies.append(accs)
		ss_accuracies.append(ss_accs)
		tr_accuracies.append(tr_accs)

		subject_numb.append(test_index[0])
		inferenceTime += inf_time

		i +=1

	print('saved on the results')

	inferenceTime = inferenceTime / i

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
		f.write('\n')
		f.write('subject_numb ')
		for item in subject_numb:
			f.write("%s " % item)
		f.write('\n')

		f.write('spectrogram time %s' % spectrogramTime)
		f.write('\n')
		f.write('inference time %s' % inferenceTime)
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