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
from utils import *
from networks import *
from dataset import EnableDataset
from itertools import combinations
import argparse


def run_classifier(mode='bilateral',classifier='CNN',sensor=["imu","emg","goin"],NN_model=None):
	"""
	Main function runs training and testing of neural network models (LIR-Net, RESNET18).
	This code runs subject independent configuration. 

	Input: argument passes through argparse. Each argument is described
	in the --help of each arguments.
	Output: No return, but generates a .txt file results of testing
	including accuracy of the models.
	"""
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
	        		'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_subjects.pth'

	if MODE_SPECIFIC_BOOL:
		SAVE_NAME= './checkpoints/'+CLASSIFIER+'/'+CLASSIFIER + '_mode_specific'+ '_'+MODE+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_subjects.pkl'
	else:
		SAVE_NAME= './checkpoints/'+CLASSIFIER+'/'+CLASSIFIER +'_'+MODE+'_'+sensor_str+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_subjects.pkl'

	if MODE_SPECIFIC_BOOL:
		RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER + NN_model+'_mode_specific'+'_'+MODE+'_'+sensor_str+'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_subjects_accuracy.txt'
	else:
		RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER + NN_model+'_'+MODE+'_'+sensor_str+'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'_BAND'+str(BAND)+'_HOP'+str(HOP)+'_subjects_accuracy.txt'


	subjects = ['156','185','186','188','189','190', '191', '192', '193', '194']

	spectrogramTime = 0.0


	# Load the dataset and train, val, test splits
	print("Loading datasets...")
	if args.data_saving:
		subject_data = []
		for subject in subjects:
			subject_data.append(EnableDataset(subject_list= [subject],data_range=(1, 51),bands=BAND,hop_length=HOP,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE,mode_specific = MODE_SPECIFIC_BOOL))
			spectrogramTime += subject_data[-1].avgSpectrogramTime
			print("subject ID",subject, "loaded")
		save_object(subject_data,SAVE_NAME)
	else:
		with open(SAVE_NAME, 'rb') as input:
		    subject_data = pickle.load(input)

	spectrogramTime = spectrogramTime / len(subjects)

	IN_CHANNELS=subject_data[0].in_channels

	device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
	print('GPU USED?',torch.cuda.is_available())

	# Choose NN models to train/test on.
	if MODE_SPECIFIC_BOOL:
		model = Network_modespecific(IN_CHANNELS,NUMB_CLASS)
	else:
		if NN_model == 'RESNET18':
			model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True) # use resnet
			num_ftrs = model.fc.in_features
			top_layer= nn.Conv2d(IN_CHANNELS, 3, kernel_size=5, stride=1, padding=2)
			model.fc = nn.Linear(num_ftrs, NUMB_CLASS)
			model = nn.Sequential(top_layer,model)
		else:
			model = Network(IN_CHANNELS,NUMB_CLASS)

	# set NN model parameters
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	num_epoch = NUB_EPOCH

	#initialize model parameters
	init_state = copy.deepcopy(model.state_dict())
	init_state_opt = copy.deepcopy(optimizer.state_dict())

	accuracies =[]
	ss_accuracies=[]
	tr_accuracies=[]
	class_accs = [0] * NUMB_CLASS
	subject_numb = []
	class_acc_list=[]

	skf = KFold(n_splits = len(subject_data), shuffle = True)
	i = 0

	train_class=trainclass(model,optimizer,args.progressbar,device,criterion,MODEL_NAME,args)

	tests=[]
	preds=[]
	inferenceTime = 0.0

	# main training/testing loop
	for train_index, test_index in skf.split(subject_data):
		# k-fold validation
		print('training subject No.:', train_index, ' Testing subject No.:',test_index)

		model.load_state_dict(init_state)
		optimizer.load_state_dict(init_state_opt)

		train_set = [subject_data[i] for i in train_index]
		test_set = [subject_data[i] for i in test_index]
		BIO_train = torch.utils.data.ConcatDataset(train_set)
		wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))

		if MODE_SPECIFIC_BOOL:
			for batch, label, dtype, prevlabel in tqdm(wholeloader,disable=args.progressbar):
				X_train = batch
				y_train = label
				types_train = dtype
				prevlabel_train = prevlabel			
		else:
			for batch, label, dtype in tqdm(wholeloader,disable=args.progressbar):
				X_train = batch
				y_train = label
				types_train = dtype
		
		BIO_train = None
		train_set = None

		BIO_test = torch.utils.data.ConcatDataset(test_set)
		wholeloader = DataLoader(BIO_test, batch_size=len(BIO_test))

		if MODE_SPECIFIC_BOOL:
			for batch, label, dtype, prevlabel in tqdm(wholeloader,disable=args.progressbar):
				X_test = batch
				y_test = label
				types_test = dtype
				prevlabel_test = prevlabel
		else:
			for batch, label, dtype in tqdm(wholeloader,disable=args.progressbar):
				X_test = batch
				y_test = label
				types_test = dtype

		BIO_test = None
		test_set = None

		if MODE_SPECIFIC_BOOL:
			onehot_train, onehot_test = prevlabel_train, prevlabel_test
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

		# append results
		accuracies.append(accs)
		ss_accuracies.append(ss_accs)
		tr_accuracies.append(tr_accs)
		preds.extend(pred)
		tests.extend(test)
		class_acc_list.append(class_acc)
		inferenceTime += inf_time
		subject_numb.append(test_index[0])

		del  test_dataset, train_dataset, trainloader, testloader

		i +=1

	# Write results to text files
	print('writing...')
	with open(RESULT_NAME, 'w') as f:
		f.write('subject_numb ')
		for item in subject_numb:
			f.write("%s " % item)
		f.write('\n')
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

	with open('./results/'+args.classifiers+'_'+sensor_str+'_'+args.laterality+'_'+'confusion_subjects.txt', 'w') as f:
		for items in confusion:
			for item in items:
				f.write("%s " % item)

			f.write('\n')

	f.close()
	print('result saved in', RESULT_NAME)


"""This block parses command line arguments and runs the main code"""
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

p.add_argument("--show_progress", dest='progressbar', action='store_false', help="show tqdm progress bar")
p.add_argument("--data_skip", dest='data_saving', action='store_false', help="skip the dataset saving/loading")
p.set_defaults(mode_specific=False)
p.set_defaults(data_saving=True)
p.set_defaults(progressbar=True)
p.set_defaults(val_on=False)

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
		print("Data Divison: subject independent splits")

		run_classifier(args)


