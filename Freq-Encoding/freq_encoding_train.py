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

	INPUT_NUM=BIO_train.input_numb
	
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

	skf = KFold(n_splits = numfolds, shuffle = True)
	i = 0


	train_class=trainclass(model,optimizer,DATA_LOAD_BOOL,device,criterion,MODEL_NAME)

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

	# 	print("######################Fold:{}#####################3".format(i+1))
	# 	train_class.train(trainloader,num_epoch)

	# 	model.load_state_dict(torch.load(MODEL_NAME))

	# 	# print("Evaluate on test set")
	# 	accs, class_acc =train_class.evaluate(testloader)
	# 	accuracies.append(accs)
	# 	for i in range(len(class_accs)):
	# 		class_accs[i] += class_acc[i]


	# 	i +=1

	print('saved on the results')
	print("average:")
	for i in range(len(class_accs)):
		if class_accs[i] == 0:
			print("Class {} has no samples".format(i))
		else:
			print("Class {} accuracy: {}".format(i, class_accs[i]/numfolds))


	model.load_state_dict(torch.load('./models/bestmodel_BATCH_SIZE32_LR1e-05_WD0.001_EPOCH200_BAND10_HOP10.pth', map_location='cpu'))

	# This is to see the activation map for the two conv layers:
	conv1 = model._modules.get('sclayer1') # Get the layers we want to hook
	conv2 = model._modules.get('sclayer2')

	act_map1 = SaveFeatures(conv1) # Setup hook, data storage
	act_map2 = SaveFeatures(conv2)

	prediction = model(BIO_train[15][0].unsqueeze(0)) # Make a prediction
	pred_probabilities = F.softmax(prediction).data.squeeze()
	act_map1.remove() # Unhook
	act_map2.remove() # Unhook

	# Save activations
	act_map1.plot_activation("conv1_activation")
	act_map1.plot_activation("conv2_activation")

	# Save one channel from the first datum in the dataset
	img = BIO_train[0][0][0]
	# img = Image.fromarray(BIO_train[15][0].numpy()[0], 'L')
	plt.imshow(img)
	plt.show()
	img.save("input.png")

	with open(RESULT_NAME, 'w') as f:
		for item in accuracies:
			f.write("%s\n" % item)
	f.close()

# Code for the different subgroups
# classifiers=['CNN']
sensors=["imu","emg","goin"]
# modes = ['bilateral', 'ipsilateral', 'contralateral']
# for classifier in classifiers:
# 	for i in range(1,4):
# 		for combo in combinations(sensors,i):
# 			sensor = [item for item in combo]
# 			for mode in modes:
# 				print(classifier, sensor, mode)
# 				run_classifier(mode=mode,classifier=classifier,sensor=sensor)

# A test using all the data
run_classifier(mode='bilateral',classifier='CNN',sensor=sensors)