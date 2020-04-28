import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset, SubsetRandomSampler



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


from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV


# torch.manual_seed(0)

def run_classifier(mode='bilateral',classifier='CNN',sensor=["imu","emg","goin"],NN_model=None, LEARNING_RATE=1e-4,WEIGHT_DECAY =1e-3):

	########## SETTINGS  ########################

	BATCH_SIZE = 32
	# LEARNING_RATE = 1e-5
	# WEIGHT_DECAY = 1e-3
	NUMB_CLASS = 5
	NUB_EPOCH= 100
	numfolds = 1
	DATA_LOAD_BOOL = True
	SAVING_BOOL = False
	HOP = 0;
	BAND = 0;
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


	model = LSTM(n_channels=INPUT_NUM,n_classes=NUMB_CLASS,gpubool=GPU_BOOL)
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

	accuracies =[]

	ss_accuracies=[]
	tr_accuracies=[]


	# class_accs = [0] * NUMB_CLASS
	class_acc_list=[]


	# skf = KFold(n_splits = numfolds, shuffle = True)
	i = 0


	train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME,BATCH_SIZE)

	tests=[]
	preds=[]


   ### splitting


	torch.manual_seed(0)

	train_size = int(0.8 * len(BIO_train))+1
	test_size = int((len(BIO_train) - train_size)/2)
	# test_size = int((len(BIO_train)-train_size))
	train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(BIO_train, [train_size, test_size, test_size])
	# train_dataset, test_dataset= torch.utils.data.random_split(BIO_train, [train_size, test_size])

	# print(test_dataset[11][0])
	trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
	testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,drop_last=True)
	valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,drop_last=True)



	train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME,BATCH_SIZE)





	# train_LSTM(trainloader,valloader,num_epoch)
	train_class.train_LSTM(trainloader,valloader,num_epoch)

	model.load_state_dict(torch.load(MODEL_NAME))

	print("############final test##########\n")
	accs,ss_accs,tr_accs,pred,test,class_acc=train_class.evaluate_LSTM(testloader,True)

	accuracies.append(accs)
	ss_accuracies.append(ss_accs)
	tr_accuracies.append(tr_accs)

	preds.extend(pred)
	tests.extend(test)

	class_acc_list.append(class_acc)


	return accs



classifiers=['LSTM']
sensors=["imu","emg","goin"]
sensor_str='_'.join(sensors)
# modes = ['bilateral','ipsilateral','contralateral']
modes = ['bilateral']


i = 3
for combo in combinations(sensors,i):
	sensor = [item for item in combo]


print(classifiers, sensor, modes)

# lrs = [1e-3,1e-4,1e-5]
lrs = [1e-5]
# wds = [1e-2,1e-3,1e-4]
wds = [1e-4,1e-5,1e-6]
val_accs = [[None]*len(lrs) for _ in range(len(wds))] 

for ind_lr,lr in enumerate(lrs):
	for ind_wd, wd in enumerate(wds):
		print("lr", lr)
		print("wd" ,wd)
		val_acc=run_classifier(mode=modes[0],classifier=classifiers[0],sensor=sensor,NN_model=None,LEARNING_RATE=lr,WEIGHT_DECAY=wd)
		# val_accs.append(val_acc)
		val_accs[ind_lr][ind_wd] = val_acc 




with open('./results/'+classifiers[0]+'/gridsearch3.txt', 'w') as f:
	f.write('lr  ')
	f.write('wd  ')
	f.write('val_acc  ')
	f.write("\n")
	for i in range(len(lrs)):
		for j in range(len(wds)):
			f.write("%s " %lrs[i])
			f.write("%s " %wds[j])
			f.write("%s " %val_accs[i][j])
			f.write("\n")
			print(lrs[i])
			print(wds[j])
			print(val_accs[i][j])
			print("\n")


f.close()


