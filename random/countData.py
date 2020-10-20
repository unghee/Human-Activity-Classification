#this file implements the mode specific and random guesser baseline classifiers.
import sys
import os
sys.path.append('.')
sys.path.append('../')
import numpy as np
from tqdm import tqdm 
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset
from dataset import EnableDataset
from sklearn.model_selection import KFold, StratifiedKFold,ShuffleSplit ,train_test_split

from utils import *
from networks import *

#parameters
numfolds = 10
DATA_LOAD_BOOL = True
MODE = 'bilateral'
CLASSIFIER = 'Random_modespecific'
SENSOR = ["imu","emg","goin"]

#load the whole dataset
BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 51),
						 bands=10,hop_length=10,mode_specific = True,model_type=CLASSIFIER,sensors=SENSOR,mode=MODE)


#count number of occurences of data, based on previous walking mode and next walking mode
vals = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

for img, labels,trigger,_ in BIO_train:

    vals[int(trigger)-1][int(labels)-1]+=1

vals = np.array(vals)


#mode specific classifier baseline: assume knowledge of previous mode
if CLASSIFIER =='Random_modespecific':
	overall_acc= np.sum(np.max(vals,0))/np.sum(vals)
	print("Random mode specific error: ", overall_acc)
	if np.max(vals).all() == np.diag(vals).all():
		ss_acc = 1
		tr_acc = 0 

#random guesser baseline: predict based on distribution of samples
elif CLASSIFIER =="Random":
	overall_acc= np.sum(vals[:,0])/np.sum(vals)
	ss_acc = vals[0][0]/np.sum(np.diag(vals))
	tr_acc = np.sum(vals[1:,0])/(np.sum(vals)-np.sum(np.diag(vals)))


	print('overall.{}, ss.{}, tr,{}'.format(overall_acc,ss_acc,tr_acc))


del vals

#load training dataset
wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))

#split dataset by number of folds
for batch, label, trigger,dtype in tqdm(wholeloader,disable=DATA_LOAD_BOOL):
	X = batch
	y = label
	tri = trigger
	types = dtype

	skf = KFold(n_splits = numfolds, shuffle = True)
	i = 0

overall_accs = []
ss_accs = []
tr_accs = []

#validate classifier through k-fold cross validation 
for train_index, test_index in skf.split(X, y):
	train_vals = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
	test_vals = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

	print("######################Fold:{}#####################".format(i+1))

	#split dataset
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	trigger_train, trigger_test = tri[train_index], tri[test_index]
	types_train, types_test = types[train_index], types[test_index]

	train_dataset = TensorDataset( X_train, y_train, trigger_train)
	test_dataset = TensorDataset( X_test, y_test, trigger_test)

	#get distribution based on fold
	for img, labels, trigger in train_dataset:
		train_vals[int(trigger)-1][int(labels)-1]+=1

	for img, labels, trigger in test_dataset:
		test_vals[int(trigger)-1][int(labels)-1]+=1

	#print(test_vals)
	test_vals=np.array(test_vals)
	train_vals=np.array(train_vals)

	if CLASSIFIER =='Random_modespecific':

		if np.argmax(train_vals,1).all() == np.array([0,1,2,3,4]).all():

			overall_acc= np.sum(np.max(test_vals,0))/np.sum(test_vals)
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


#write results
if not os.path.exists('./results/'+CLASSIFIER):
		os.makedirs('./results/'+CLASSIFIER)
RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER + '_accuracy.txt'
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

