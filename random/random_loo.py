import sys
import os
sys.path.append('.')
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset
import pickle
from sklearn.model_selection import KFold, StratifiedKFold,ShuffleSplit ,train_test_split
import argparse
from itertools import combinations
from dataset import EnableDataset
from utils import *
from networks import *


def run_classifier(args):
	"""
	Main function runs training and testing of Random classifier.
	This code runs subject independent configuration. 

	Input: argument passes through argparse. Each argument is described
	in the --help of each arguments.
	Output: No return, but generates a .txt file results of testing
	including accuracy of the models.
	"""
	#parameters
	numfolds = 10
	SAVING_BOOL = args.data_saving
	MODE = args.laterality
	CLASSIFIER = args.classifiers
	SENSOR = args.sensors

	subjects = ['156','185','186','188','189','190', '191', '192', '193', '194']

	#save data for ease of later use
	if SAVING_BOOL:
		if not os.path.exists('./checkpoints/'):
			os.makedirs('./checkpoints/')
		subject_data = []
		for subject in subjects:
			subject_data.append(EnableDataset(subject_list= [subject],model_type=CLASSIFIER,sensors=SENSOR,mode=MODE))
		save_object(subject_data,'./checkpoints/count_Data_features.pkl')	
	else:
		with open('./checkpoints/count_Data_features.pkl', 'rb') as input:
			   subject_data = pickle.load(input)

	skf = KFold(n_splits = numfolds, shuffle = True)
	i = 0

	overall_accs = []
	ss_accs = []
	tr_accs = []
	#run leave-one-out evaluation of random guesser and mode specific classifiers
	for train_index, test_index in skf.split(subject_data):
		train_vals = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
		test_vals = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

		print("######################Fold:{}#####################".format(i+1))

		#split data based on leaving one subject out
		train_set = [subject_data[i] for i in train_index]
		test_set = [subject_data[i] for i in test_index]
		BIO_train = torch.utils.data.ConcatDataset(train_set)
		wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))
		for batch, label, trigger,dtype in wholeloader:
			X_train = batch
			y_train = label
			types_train = dtype
			trigger_train = trigger

		BIO_test = torch.utils.data.ConcatDataset(test_set)
		wholeloader = DataLoader(BIO_test, batch_size=len(BIO_train))
		for batch, label, trigger,dtype in wholeloader:
			X_test = batch
			y_test = label
			types_test = dtype
			trigger_test = trigger


		train_dataset = TensorDataset( X_train, y_train, trigger_train)
		test_dataset = TensorDataset( X_test, y_test, trigger_test)

		#get dataset statistics
		for img, labels, trigger in train_dataset:
			train_vals[int(trigger)-1][int(labels)-1]+=1

		for img, labels, trigger in test_dataset:
			test_vals[int(trigger)-1][int(labels)-1]+=1

		test_vals=np.array(test_vals)
		train_vals=np.array(train_vals)

		#evaluate mode specific classifier
		if args.mode_specific:
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

		#evaluate random guesser
		else:
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

	#save results
	print('writing...')
	RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER + '_subjects_accuracy.txt'
	if not os.path.exists('./results/'+CLASSIFIER):
			os.makedirs('./results/'+CLASSIFIER)
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

"""This block parses command line arguments and runs the main code"""
p = argparse.ArgumentParser()
p.add_argument("--classifiers", default='Random', help="classifier types: 'Random'")
p.add_argument("--sensors", nargs="+", default=["imu","emg","gon"], help="select combinations of sensor modality types: img, emg, gonio")
p.add_argument("--all_comb", dest='all_comb', action='store_true', help="loop through all combinations")
p.add_argument("--laterality", default='bilateral', type=str, help="select laterality types, bilateral, ipsilateral, contralateral")
p.add_argument("--data_skip", dest='data_saving', action='store_false', help="skip the dataset saving/loading")
p.add_argument("--mode_specific", action='store_true', help="mode specific configuration")

args = p.parse_args()

p.set_defaults(data_saving=True)
p.set_defaults(all_comb=False)
p.set_defaults(mode_specific=False)

comb_number = len(args.sensors)

if args.all_comb:
	print('looping through all combinations, overriding sensor selection')
	args.sensors = ["imu","emg","gon"]
	comb_number = 1

for i in range(comb_number,4):
	print('Number of sensors range:' , i ,'to',len(args.sensors))
	for combo in combinations(args.sensors,i):
		sensor = [item for item in combo]
		print("Classifer type: ", args.classifiers)
		print("Sensor modality: ", sensor)
		print("Sensor laterality: ", args.laterality)

		run_classifier(args)