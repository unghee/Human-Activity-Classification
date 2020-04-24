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



def run_classifier(mode='bilateral',classifier='CNN',sensor=["imu","emg","goin"],NN_model=None):

	########## SETTINGS  ########################

	BATCH_SIZE = 32
	LEARNING_RATE = 1e-5
	WEIGHT_DECAY = 1e-3
	NUMB_CLASS = 5
	NUB_EPOCH= 2
	numfolds = 10
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


	skf = KFold(n_splits = numfolds, shuffle = True)
	i = 0


	train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME)

	tests=[]
	preds=[]

	train_size = int(0.8 * len(BIO_train))+1
	test_size = int((len(BIO_train) - train_size)/2)
	train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(BIO_train, [train_size, test_size, test_size])

	trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
	valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


	def train_LSTM(loader,valloader, num_epoch = 20,onehot=None): # Train the model
	    loss_history=[]
	    val_history=[]
	    print("Start training...")
	    model.train()

	    pre_loss=10000
	    for i in range(num_epoch):
	        running_loss = []
	        h = model.init_hidden(BATCH_SIZE)  
	        for batch, label, types in tqdm(loader,disable=DATA_LOAD_BOOL):
	        	batch = batch.to(device)
	        	label = label.to(device)
	        	label = label -1 # indexing start from 1 (removing sitting conditon)
	        	h = tuple([e.data for e in h])
	        	# h = tuple([each.repeat(1, BATCH_SIZE, 1).data for each in h])
	        	optimizer.zero_grad()
	        	pred,h = model(batch,h,BATCH_SIZE)
	        	loss = criterion(pred, label.long())
	        	running_loss.append(loss.item())
	        	loss.backward()
	        	optimizer.step()
	        loss_mean= np.mean(running_loss)
	        loss_history.append(loss_mean)
	        val_acc = evaluate_LSTM(model, valloader)
	        if loss_mean< pre_loss:
	        	pre_loss = loss_mean
	        	torch.save(model.state_dict(), MODEL_NAME)
	        	print("*model saved*")
	        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc))
	    print("Done!")
	    return loss_history, val_history


	def evaluate_LSTM(loader):
	    model.eval()
	    correct = 0
	    steady_state_correct = 0
	    tot_steady_state = 0
	    transitional_correct = 0
	    tot_transitional = 0
	    preds=[]
	    tests=[]

	    class_correct = [0]*6
	    class_total = [0]*6
	    class_acc=[]

	    h = model.init_hidden(BATCH_SIZE) 

	    with torch.no_grad():
	        count = 0
	        totalloss = 0
	        for batch, label, types in tqdm(loader,disable=DATA_LOAD_BOOL):
	            batch = batch.to(device)
	            label = label-1 # indexing start from 1 (removing sitting conditon)
	            label = label.to(device)
	            h = tuple([each.data for each in h])

	            pred,h = model(batch,h,BATCH_SIZE)
	            totalloss += criterion(pred, label)
	            count +=1
	            preds.extend((torch.argmax(pred,dim=1)).tolist())
	            tests.extend(label.tolist())

	            correct += (torch.argmax(pred,dim=1)==label).sum().item()
	            steady_state_correct += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), types == 1)).sum().item()
	            tot_steady_state += (types == 1).sum().item()
	            transitional_correct += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), types == 0)).sum().item()
	            tot_transitional += (types == 0).sum().item()

	            for i in range(len(class_correct)):
	                class_correct[i] += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), label.cpu() == i)).sum().item()
	                class_total[i] += (label == i).sum().item()
	    acc = correct/len(loader.dataset)
	    for i in range(len(class_correct)):
	    	if class_total[i] == 0:
	    		print("Class {} has no samples".format(i))
	    	else:
	    		print("Class {} accuracy: {}".format(i, class_correct[i]/class_total[i]))
	    		class_acc.append(class_correct[i]/class_total[i])
	    ss_acc = steady_state_correct/tot_steady_state if tot_steady_state != 0 else "No steady state samples used"
	    tr_acc = transitional_correct/tot_transitional if tot_transitional != 0 else "No transitional samples used"
	    print("Evaluation loss: {}".format(totalloss/count))
	    print("Evaluation accuracy: {}".format(acc))
	    print("Steady-state accuracy: {}".format(ss_acc))
	    print("Transistional accuracy: {}".format(tr_acc))






	train_LSTM(trainloader,valloader,num_epoch)
	model.load_state_dict(torch.load(MODEL_NAME))
	accs,ss_accs,tr_accs,pred,test,class_acc=evaluate_LSTM(testloader)

	accuracies.append(accs)
	ss_accuracies.append(ss_accs)
	tr_accuracies.append(tr_accs)

	preds.extend(pred)
	tests.extend(test)

	class_acc_list.append(class_acc)








	# for train_index, test_index in skf.split(X, y, types):

	# 	model.load_state_dict(init_state)
	# 	optimizer.load_state_dict(init_state_opt)

	# 	X_train, X_test = X[train_index], X[test_index]
	# 	y_train, y_test = y[train_index], y[test_index]
	# 	types_train, types_test = types[train_index], types[test_index]

	# 	train_dataset = TensorDataset( X_train, y_train, types_train)
	# 	test_dataset = TensorDataset( X_test, y_test, types_test)

	# 	trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	# 	testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

	# 	print("######################Fold:{}#####################".format(i+1))
	# 	train_class.train(trainloader,num_epoch)

	# 	model.load_state_dict(torch.load(MODEL_NAME))


	# 	# print("Evaluate on test set")

	# 	accs,ss_accs,tr_accs,pred,test,class_acc=train_class.evaluate(testloader)
	# 	accuracies.append(accs)
	# 	ss_accuracies.append(ss_accs)
	# 	tr_accuracies.append(tr_accs)

	# 	preds.extend(pred)
	# 	tests.extend(test)

	# 	class_acc_list.append(class_acc)

	# 	i +=1

	print('saved on the results')


	# model.load_state_dict(torch.load('./models/bestmodel_BATCH_SIZE32_LR1e-05_WD0.001_EPOCH200_BAND10_HOP10.pth', map_location='cpu'))


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

	conf= confusion_matrix(tests, preds)
	print(conf)
	print(classification_report(tests, preds, digits=3))

	return conf



classifiers=['LSTM']
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

with open('./results/'+classifiers[0]+'_'+sensor_str+'_'+modes[0]+'_'+'confusion.txt', 'w') as f:
	for items in confusion:
		for item in items:
			f.write("%s " % item)

		f.write('\n')
f.close()


