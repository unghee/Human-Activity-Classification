import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from tqdm import tqdm # Displays a progress bar
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA, sparse_encode
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from itertools import combinations
import pickle
import argparse
import sys,os
sys.path.append('.')
from utils import *
from dataset import EnableDataset

def run_classifier(args):
	"""
	Main function runs training and testing of Heuristic based machine
	learning models (SVM, LDA)

	Input: argument passes through argparse. Each argument is described
	in the --help of each arguments.
	Output: No return, but generates a .txt file results of testing
	including accuracy of the models.
	"""
	########## PRAMETER SETTINGS  ###############
	MODE = args.laterality
	CLASSIFIER = args.classifiers
	SENSOR = args.sensors
	############################################

	sensor_str='_'.join(SENSOR)

	RESULT_NAME= './results/' + CLASSIFIER +'/'+CLASSIFIER+'_'+MODE+'_'+sensor_str+'_accuracy_nway_subject.txt'

	SAVE_NAME= './checkpoints/mode_specfic'+'_'+MODE+'_'+sensor_str+'_subject.pkl'

	if not os.path.exists('./results/mode_specific'):
		os.makedirs('./results/mode_specific')

	if not os.path.exists('./checkpoints/'+CLASSIFIER):
			os.makedirs('./checkpoints/'+CLASSIFIER)

	numb_class= [5,2,2,2,2] # Define output class number for each mode classifiers 
	len_class = len(numb_class)
	len_phase = 4
	BIO_trains=[]

	BIO_trains_len=0

	k = 0
	subjects = ['156','185','186','188','189','190', '191', '192', '193', '194']

	# Loading/saving the ENABL3S dataset
	if args.data_saving:
		for i in range(1,len_class+1):
			for j in range(1,len_phase+1):
				subject_data = []
				for subject in subjects:
					subject_data.append(EnableDataset(subject_list= [subject],phaselabel=j,prevlabel=i,model_type='LDA',sensors=SENSOR,mode=MODE))
				BIO_trains.append(subject_data)
				k +=1
				print(k)
		save_object(BIO_trains,SAVE_NAME)

	else:	
		with open(SAVE_NAME, 'rb') as input:
			 BIO_trains = pickle.load(input)

	models=[]
	tot=0
	correct=0
	steady_state_correct = 0
	tot_steady_state = 0
	transitional_correct = 0
	tot_transitional = 0
	for i in range(1,len_class+1):
		for j in range(1,len_phase+1):
			if CLASSIFIER == 'LDA':
				model = LinearDiscriminantAnalysis()
			elif CLASSIFIER == 'SVM':
				model = SVC(kernel = 'linear', C = 10)
			models.append(model)

	k =0

	# initialize variables for model performance metric
	accuracies=[[[] for x in range(len_phase)]for y in range(len_class)]
	ss_accuracies=[[[] for x in range(len_phase)]for y in range(len_class)]
	tr_accuracies=[[[] for x in range(len_phase)]for y in range(len_class)]

	subject_numb = []

	tot_numb_mat = np.zeros((10,20))
	ss_numb_mat = np.zeros((10,20))
	tr_numb_mat = np.zeros((10,20))
	tot_mat = np.zeros((10,20))
	ss_mat = np.zeros((10,20))
	tr_mat = np.zeros((10,20))

	# Define cross-validation parameters
	numfolds = 10
	skf = KFold(n_splits = numfolds, shuffle = True)

	data_lens=0

	"""
	main testing/training loop of mode-specific classifiers. 
	Separate classifiers for each activity classes (LW,RA,RD,SA,SD) 
	and phase (Right left toe off/heel contact)
	"""
	for i in range(1, len_class+1):
		for j in range(1,len_phase+1):
			print("**************mode #", i, "****phase", j)

			scale = preprocessing.StandardScaler()
			pca = PCA()
			scale_PCA = Pipeline([('norm',scale),('dimred',pca)])

			m = 0

			for train_index, test_index in skf.split(BIO_trains[k]):

				subject_data=BIO_trains[k] 
				train_set = [subject_data[i] for i in train_index]
				test_set = [subject_data[i] for i in test_index]
				BIO_train = torch.utils.data.ConcatDataset(train_set)
				wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))

				for batch, label, dtype in tqdm(wholeloader, disable=args.progressbar):
					X_train = batch
					y_train = label
					types_train = dtype

				BIO_train = None
				train_set = None

				BIO_test = torch.utils.data.ConcatDataset(test_set)
				wholeloader = DataLoader(BIO_test, batch_size=len(BIO_test))

				for batch, label, dtype in tqdm(wholeloader, disable=args.progressbar):
					X_test = batch
					y_test = label
					types_test = dtype

				BIO_test = None
				test_set = None

				data_lens += len(X_train) + len(X_test)

				if CLASSIFIER == 'LDA':
					scale_PCA.fit(X_train)

					feats_train_PCA = scale_PCA.transform(X_train)
					feats_test_PCA = scale_PCA.transform(X_test)

					pcaexplainedvar = np.cumsum(scale_PCA.named_steps['dimred'].explained_variance_ratio_)
					pcanumcomps = min(min(np.where(pcaexplainedvar > 0.95))) + 1

					unique_modes = np.unique(y_train)
					models[k].set_params(priors = np.ones(len(unique_modes))/len(unique_modes))

					models[k].fit(feats_train_PCA[:,0:pcanumcomps],y_train)
					y_pred=models[k].predict(feats_test_PCA[:,0:pcanumcomps]).ravel()

				elif CLASSIFIER == 'SVM':
					scale.fit(X_train)

					feats_train_norm = scale.transform(X_train)
					feats_test_norm = scale.transform(X_test )

					models[k].fit(feats_train_norm,y_train)
					y_pred=models[k].predict(feats_test_norm)

				correct = (y_pred==np.array(y_test)).sum().item()
				tot = len(y_test)
				steady_state_correct = (np.logical_and(y_pred==np.array(y_test), types_test == 1)).sum().item()
				tot_steady_state = (types_test == 1).sum().item()
				transitional_correct = (np.logical_and(y_pred==np.array(y_test), types_test == 0)).sum().item()
				tot_transitional = (types_test == 0).sum().item()

				tot_acc = correct/tot
				ss_acc = steady_state_correct/tot_steady_state if tot_steady_state != 0 else "No steady state samples used"
				tr_acc = transitional_correct/tot_transitional if tot_transitional != 0 else "No transitional samples used"

				tot_numb_mat[m,k] = tot
				ss_numb_mat[m,k] = tot_steady_state
				tr_numb_mat[m,k] = tot_transitional

				tot_mat[m,k] = correct
				ss_mat[m,k] = steady_state_correct
				tr_mat[m,k] = transitional_correct


				m +=1
			k +=1
			del pca, X_train, X_test, y_train, y_test


	tot_numbs=np.sum(tot_numb_mat,axis=1)
	ss_numbs=np.sum(ss_numb_mat,axis=1)
	tr_numbs=np.sum(tr_numb_mat,axis=1)

	accuracies=np.sum(tot_mat,axis=1)/tot_numbs
	ss_accuracies=np.sum(ss_mat,axis=1)/ss_numbs
	tr_accuracies=np.sum(tr_mat,axis=1)/tr_numbs

	print('total number of classifiers: ' ,k)
	print('total number of data: ' ,data_lens )
	print('Accuracy_total:', np.mean(accuracies))
	print('Steady-state:', np.mean(ss_accuracies) )
	print('Transistional:', np.mean(tr_accuracies))

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
	f.close()


"""This block parses command line arguments and runs the main code"""
p = argparse.ArgumentParser()
p.add_argument("--classifiers", default="LDA", help="classifier types: LDA, SVM")
p.add_argument("--sensors", nargs="+", default=["imu","emg","gon"], help="select combinations of sensor modality types: img, emg, gonio")
p.add_argument("--all_comb", dest='all_comb', action='store_true', help="loop through all combinations")
p.add_argument("--laterality", default='bilateral', type=str, help="select laterality types, bilateral, ipsilateral, contralateral")
p.add_argument("--data_skip", dest='data_saving', action='store_false', help="skip the dataset saving/loading")
p.add_argument("--show_progress", dest='progressbar', action='store_false', help="show tqdm progress bar")

args = p.parse_args()

p.set_defaults(data_saving=True)
p.set_defaults(all_comb=False)
p.set_defaults(progressbar=True)

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