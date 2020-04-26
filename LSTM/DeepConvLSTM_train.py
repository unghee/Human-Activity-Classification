
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
from sliding_window import *
import time

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def run_classifier(mode='bilateral',classifier='CNN',sensor=["imu","emg","goin"],NN_model=None):

	########## SETTINGS  ########################

	# BATCH_SIZE = 32
	BATCH_SIZE = 32
	# LEARNING_RATE = 1e-5
	LEARNING_RATE = 1e-4
	# WEIGHT_DECAY = 1e-3
	WEIGHT_DECAY=1e-4
	NUMB_CLASS = 5
	NUB_EPOCH= 200
	numfolds = 2
	DATA_LOAD_BOOL = True
	SAVING_BOOL = True
	HOP = 0;
	BAND = 0;

	SLIDING_WINDOW_LENGTH= 500
	SLIDING_WINDOW_STEP = 250
	############################################



	MODE = mode
	CLASSIFIER = classifier
	SENSOR = sensor
	sensor_str='_'.join(SENSOR)


	MODEL_NAME = './models/LSTM/bestmodel'+ \
	        		'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'.pth'

	# RESULT_NAME= './results/Freq-Encoding/accuracy'+ \
	        		# '_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'.txt'


	RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER +'_'+MODE+'_'+sensor_str+'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_accuracy.txt'

	SAVE_NAME= './checkpoints/'+CLASSIFIER+'/'+CLASSIFIER +'_'+MODE+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'.pkl'

	if not os.path.exists('./models/'+CLASSIFIER):
		os.makedirs('./models/'+CLASSIFIER)


	if not os.path.exists('./results/'+CLASSIFIER):
		os.makedirs('./results/'+CLASSIFIER)

	if not os.path.exists('./checkpoints/'+CLASSIFIER):
		os.makedirs('./checkpoints/'+CLASSIFIER)


	if SAVING_BOOL:

		# Load the dataset and train, val, test splits
		print("Loading datasets...")

		BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 51),time_series=True,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE)

		save_object(BIO_train,SAVE_NAME)

	with open(SAVE_NAME, 'rb') as input:
	    BIO_train = pickle.load(input)

	INPUT_NUM=BIO_train.input_numb

	wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))


	device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
	# device = "cpu"
	print('GPU USED?',torch.cuda.is_available())
	GPU_BOOL=torch.cuda.is_available()
	# GPU_BOOL =False


	# model = LSTM(n_channels=INPUT_NUM,n_classes=NUMB_CLASS,gpubool=GPU_BOOL)

	model = DeepConvLSTM(n_channels=INPUT_NUM,n_classes=NUMB_CLASS,gpubool=GPU_BOOL)


	model.apply(init_weights)


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

	X=X.permute(0,2,1)

	### SLIDING WINDOW

	X_train, y_train = opp_sliding_window(X, y, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
	# X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)








	###

	accuracies =[]

	ss_accuracies=[]
	tr_accuracies=[]


	# class_accs = [0] * NUMB_CLASS
	class_acc_list=[]




	train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME,BATCH_SIZE)

	tests=[]
	preds=[]




	skf = KFold(n_splits = numfolds, shuffle = True)

	#####################33 do for loop and use python extend
	i = 0

	for train_index, test_index in skf.split(X, y, types):

		model.load_state_dict(init_state)
		optimizer.load_state_dict(init_state_opt)

		# X_train, X_test = [0]*len(train_index), [0]*len(test_index)
		# X_train=torch.Tensor([]).double()
		# X_train=[]
		# y_train= []
		# for j in train_index:
		# 	# X_train=torch.cat((X_train,X[j]))
		# 	X_train.append(X[j]) 
		X_train = torch.ones((len(train_index),X.size(1),X.size(2)))
		y_train = torch.ones((len(train_index)))
		X_test = torch.ones((len(test_index),X.size(1),X.size(2)))
		y_test = torch.ones((len(test_index)))

		m = 0
		for j in train_index:
			X_train[m]=X[j]
			y_train[m]=y[j]
			m +=1

		l = 0
		time.sleep(1)
		for k in test_index:
			X_test[l]=X[k]
			y_test[l]=y[k]
			l +=1

		# X_train, X_test = X[train_index], X[test_index]
		# y_train, y_test = y[train_index], y[test_index]
		types_train, types_test = types[train_index], types[test_index]

		train_dataset = TensorDataset( X_train, y_train, types_train)
		test_dataset = TensorDataset( X_test, y_test, types_test)

		trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
		testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,drop_last=True)

		print("######################Fold:{}#####################".format(i+1))
		train_class.train_LSTM(loader=trainloader,num_epoch=num_epoch)

		model.load_state_dict(torch.load(MODEL_NAME))

		accs,ss_accs,tr_accs,pred,test,class_acc=train_class.evaluate_LSTM(testloader)
		accuracies.append(accs)
		ss_accuracies.append(ss_accs)
		tr_accuracies.append(tr_accs)

		preds.extend(pred)
		tests.extend(test)

		class_acc_list.append(class_acc)

		del X_train, y_train, X_test, y_test, types_train, types_test, train_dataset, test_dataset , trainloader, testloader

	

		i +=1

	print('saved on the results')


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

		for j in range(0,5):
			f.write('\n')
			f.write('class {} '.format(j))
			for m in range(0,numfolds):
				f.write("%s " % class_acc_list[m][j])



	f.close()

	# conf= confusion_matrix(tests, preds)
	# print(conf)
	# print(classification_report(tests, preds, digits=3))

	# return conf



# classifiers=['LSTM']
classifiers=['DeepConvLSTM']

sensors=["imu","emg","goin"]
sensor_str='_'.join(sensors)
# modes = ['bilateral','ipsilateral','contralateral']
modes = ['bilateral']
# NN= 'RESNET'
# NN = 'LAPNET'
for classifier in classifiers:
	for i in range(3,4):
		for combo in combinations(sensors,i):
			sensor = [item for item in combo]
			for mode in modes:
				print(classifier, sensor, mode)
				confusion=run_classifier(mode=mode,classifier=classifier,sensor=sensor,NN_model=None)

# with open('./results/'+classifiers[0]+'_'+sensor_str+'_'+modes[0]+'_'+'confusion.txt', 'w') as f:
# 	for items in confusion:
# 		for item in items:
# 			f.write("%s " % item)

# 		f.write('\n')
# f.close()


