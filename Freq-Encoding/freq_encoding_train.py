import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset

from sklearn.model_selection import KFold, StratifiedKFold,ShuffleSplit ,train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from PIL import Image
import pickle
from itertools import combinations
import copy
import os
import random
import sys,os
sys.path.append('.')

from utils import *
from networks import *
from dataset import EnableDataset



def run_classifier(args):

	########## PRAMETER SETTINGS  ########################
	BATCH_SIZE = args.batch_size
	LEARNING_RATE = args.lr
	WEIGHT_DECAY = args.weight_decay
	NUMB_CLASS = 5
	NUB_EPOCH= args.num_epoch
	numfolds = args.num_folds
	BAND=args.band
	HOP=args.hop

	SENSOR= args.sensors
	MODE = args.laterality
	CLASSIFIER = args.classifiers
	NN_model = args.nn_architecture
	MODE_SPECIFIC_BOOL= args.mode_specific
	############################################

	sensor_str='_'.join(SENSOR)

	MODEL_NAME = './models/Freq-Encoding/bestmodel'+ \
	        		'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'.pth'

	if MODE_SPECIFIC_BOOL:
		RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER + NN_model+'_mode_specific'+'_'+MODE+'_'+sensor_str+'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_accuracy.txt'
	else:
		RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER + NN_model+'_'+MODE+'_'+sensor_str+'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_accuracy.txt'

	if MODE_SPECIFIC_BOOL:
		SAVE_NAME= './checkpoints/'+CLASSIFIER+'/'+CLASSIFIER+'_'+MODE+'_mode_specific'+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'.pkl'
	else:
		SAVE_NAME= './checkpoints/'+CLASSIFIER+'/'+CLASSIFIER +'_'+MODE+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'.pkl'

	if not os.path.exists('./models/Freq-Encoding'):
		os.makedirs('./models/Freq-Encoding')

	if not os.path.exists('./results/'+CLASSIFIER):
		os.makedirs('./results/'+CLASSIFIER)

	if not os.path.exists('./checkpoints/'+CLASSIFIER):
		os.makedirs('./checkpoints/'+CLASSIFIER)

	spectrogramTime = 0.0

	if args.data_saving:
		# Load the dataset and train, val, test splits
		print("Loading datasets...")
		BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194']  \
			,data_range=(1, 51),bands=BAND,hop_length=HOP,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE,mode_specific = MODE_SPECIFIC_BOOL)
		spectrogramTime += BIO_train.avgSpectrogramTime
		save_object(BIO_train,SAVE_NAME)
	with open(SAVE_NAME, 'rb') as input:
	    BIO_train = pickle.load(input)

	IN_CHANNELS=BIO_train.in_channels

	wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))

	device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
	print('GPU USED?',torch.cuda.is_available())

	if MODE_SPECIFIC_BOOL:
		model = Network_modespecific(IN_CHANNELS,NUMB_CLASS)
	else: 
		if NN_model == 'RESNET18':
			print("model :**** RESNET18 ****")
			model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True) # use resnet
			num_ftrs = model.fc.in_features
			top_layer= nn.Conv2d(IN_CHANNELS, 3, kernel_size=5, stride=1, padding=2)
			model.fc = nn.Linear(num_ftrs, NUMB_CLASS)
			model = nn.Sequential(top_layer,model)
		else:
			model = Network(IN_CHANNELS,NUMB_CLASS)

	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	num_epoch = NUB_EPOCH

	init_state = copy.deepcopy(model.state_dict())
	init_state_opt = copy.deepcopy(optimizer.state_dict())


	if MODE_SPECIFIC_BOOL:
		one_hot_embed= torch.eye(5)
		for batch, label, dtype, prevlabels  in tqdm(wholeloader,disable=args.progressbar):
			X = batch
			y = label
			types = dtype
			prevlabel = prevlabels

	else:
		for batch, label, dtype in tqdm(wholeloader,disable=args.progressbar):
			X = batch
			y = label
			types = dtype

	accuracies =[]
	ss_accuracies=[]
	tr_accuracies=[]
	tests=[]
	preds=[]
	inferenceTime = 0.0
	class_acc_list=[]


	if args.val_on:

		train_class=trainclass(model,optimizer,args.progressbar,device,criterion,MODEL_NAME,args)


		numfolds = 1

		train_size = int(0.8 * len(BIO_train))+1
		test_size = int((len(BIO_train) - train_size)/2)
		train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(BIO_train, [train_size, test_size, test_size])
		
		trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
		valloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
		testloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

		train_class.train(trainloader,num_epoch, valloader=valloader)
		model.load_state_dict(torch.load(MODEL_NAME))

		print("Evaluate on test set")
		accs,ss_accs,tr_accs,pred,test,class_acc,inf_time=train_class.evaluate(testloader)
		accuracies.append(accs)
		ss_accuracies.append(ss_accs)
		tr_accuracies.append(tr_accs)

		preds.extend(pred)
		tests.extend(test)

		class_acc_list.append(class_acc)

		inferenceTime += inf_time


	else:
		skf = KFold(n_splits = numfolds, shuffle = True)
		i = 0

		train_class=trainclass(model,optimizer,args.progressbar,device,criterion,MODEL_NAME,args)

		for train_index, test_index in skf.split(X, y, types):

			model.load_state_dict(init_state)
			optimizer.load_state_dict(init_state_opt)

			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			types_train, types_test = types[train_index], types[test_index]
			
			if MODE_SPECIFIC_BOOL:
				# onehot_train, onehot_test = one_hot_embed[prevlabel[train_index]], one_hot_embed[prevlabel[test_index]]
				onehot_train, onehot_test = prevlabel[train_index], prevlabel[test_index]
				train_dataset = TensorDataset( X_train, y_train, types_train,onehot_train)
				test_dataset = TensorDataset( X_test, y_test, types_test,onehot_test)
			else:
				train_dataset = TensorDataset( X_train, y_train, types_train)
				test_dataset = TensorDataset( X_test, y_test, types_test)

			trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
			testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

			print("######################Fold:{}#####################".format(i+1))
			train_class.train(trainloader,num_epoch)

			model.load_state_dict(torch.load(MODEL_NAME))

			print("Evaluate on test set")
			accs,ss_accs,tr_accs,pred,test,class_acc,inf_time=train_class.evaluate(testloader)
			accuracies.append(accs)
			ss_accuracies.append(ss_accs)
			tr_accuracies.append(tr_accs)

			preds.extend(pred)
			tests.extend(test)

			class_acc_list.append(class_acc)

			inferenceTime += inf_time

			i +=1

	print('saved on the results')

	# Write results to text files
	inferenceTime = inferenceTime/len(preds)
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
		f.write('\n')
		if args.data_saving:
			f.write('spectrogram time %s' % spectrogramTime)
			f.write('\n')
		f.write('inference time %s' % inferenceTime)

	f.close()

	confusion= confusion_matrix(tests, preds)
	print(confusion)
	print(classification_report(tests, preds, digits=3))

	with open('./results/'+args.classifiers+'_'+sensor_str+'_'+args.laterality+'_'+'confusion.txt', 'w') as f:
		for items in confusion:
			for item in items:
				f.write("%s " % item)

			f.write('\n')
	f.close()



"""This block parses command line arguments and runs the main code"""
import argparse

p = argparse.ArgumentParser()
p.add_argument("--classifiers", default="CNN", help="classifier types: CNN")
p.add_argument("--sensors", nargs="+", default=["imu","emg","gon"], help="select combinations of sensor modality types: img, emg, gonio")
p.add_argument("--all_comb", dest='all_comb', action='store_true', help="loop through all combinations")
p.add_argument("--laterality", default='bilateral', type=str, help="select laterality types, bilateral, ipsilateral, contralateral")
p.add_argument("--nn_architecture", default='LIRNET',type=str,help="select nn architectures: LIRNET, RESNET")
p.add_argument("--mode_specific", action='store_true', help="mode specific configuration")

p.add_argument("--batch_size", default=32, type=int, help="batch size")
p.add_argument("--lr", default=1e-5, type=float, help="learning rate")
p.add_argument("--weight_decay", default=1e-3, type=float, help="weight decay")
p.add_argument("--num_epoch", default=200, type=int, help="number of epochs")
p.add_argument("--num_folds", default=10, type=int, help="number of folds for cross validation")
p.add_argument("--band", default=10, type=int, help="band length for spectrogram")
p.add_argument("--hop", default=10, type=int, help="hop length for spectrogram")

p.add_argument("--nfold", dest='val_on', action='store_true', help="n-fold validation, and show validation accuracies")
p.add_argument("--show_progress", dest='progressbar', action='store_false', help="show tqdm progress bar")
p.add_argument("--data_skip", dest='data_saving', action='store_false', help="skip the dataset saving/loading")
p.set_defaults(mode_specific=False)
p.set_defaults(data_saving=True)
p.set_defaults(progressbar=True)
p.set_defaults(val_on=False)
p.set_defaults(all_comb=False)

args = p.parse_args()

comb_number = len(args.sensors)

for i in range(comb_number,4):
	print('Number of sensors range:' , i ,'to',len(args.sensors))
	for combo in combinations(args.sensors,i):
		sensor = [item for item in combo]
		print("Classifer type: ", args.classifiers)
		print("Sensor modality: ", sensor)
		print("Sensor laterality: ", args.laterality)
		if args.mode_specific:
			print("Classifier config: mode specific")
			if args.nn_architecture == "RESNET18":
				print("no modespecific available for RESNET18, changing to LIRNET..")
				args.nn_architecture = "LIRNET"
		else: 
			print("Classifier config: generic")
		if args.classifiers == "CNN":
			print("NN architecture: ",args.nn_architecture)
		if args.val_on:
			print("Data Divison: k fold validation")
		else:
			print("Data Divison: 8:1:1 split")

		run_classifier(args)




