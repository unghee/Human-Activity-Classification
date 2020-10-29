import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm # Displays a progress bar


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA, sparse_encode
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import sys,os
sys.path.append('.')
from utils import *
from dataset import EnableDataset

from itertools import combinations

def run_classifier(args):
	########## PRAMETER SETTINGS  ########################
	MODE = args.laterality
	CLASSIFIER = args.classifiers
	SENSOR = args.sensors
	############################################

	SENSOR = sensor
	sensor_str='_'.join(SENSOR)

	RESULT_NAME= './results/'+'/'+CLASSIFIER+'/'+CLASSIFIER+'_mode_specfic'+'_'+MODE+'_'+sensor_str+'_accuracy.txt'

	SAVE_NAME= './checkpoints/'+CLASSIFIER+'/mode_specfic'+'_'+MODE+'_'+sensor_str+'.pkl'

	if not os.path.exists('./results/'+CLASSIFIER):
		os.makedirs('./results/'+CLASSIFIER)

	if not os.path.exists('./checkpoints/'+CLASSIFIER):
		os.makedirs('./checkpoints/'+CLASSIFIER)

	## for N way classifieres
	numb_class= [5,2,2,2,2]
	len_class = len(numb_class)
	len_phase = 4
	BIO_trains=[]

	BIO_trains_len=0

	print("Loading datasets...")
	if args.data_saving:
		k = 0
		for i in range(1,len_class+1):
			for j in range(1,len_phase+1):
				BIO_trains.append(EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],phaselabel=j,prevlabel=i,model_type='LDA',sensors=SENSOR,mode=MODE))
				# BIO_trains_len += len(BIO_trains[k])
				k +=1
		save_object(BIO_trains,SAVE_NAME)
	else:
		with open(SAVE_NAME, 'rb') as input:
			BIO_trains = pickle.load(input)		
		
	wholeloaders = []

	k = 0
	for i in range(1,len_class+1):
		for j in range(1,len_phase+1):
			BIO_trains_len += len(BIO_trains[k])
			wholeloaders.append(DataLoader(BIO_trains[k],batch_size=len(BIO_trains[k])))
			# print(k)
			k +=1

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

	accuracies=[[[] for x in range(len_phase)]for y in range(len_class)]
	ss_accuracies=[[[] for x in range(len_phase)]for y in range(len_class)]
	tr_accuracies=[[[] for x in range(len_phase)]for y in range(len_class)]

	tot_numb_mat = np.zeros((10,20))
	ss_numb_mat = np.zeros((10,20))
	tr_numb_mat = np.zeros((10,20))

	tot_mat = np.zeros((10,20))
	ss_mat = np.zeros((10,20))
	tr_mat = np.zeros((10,20))

	# Define cross-validation parameters
	numfolds = 10
	# kf = KFold(n_splits = numfolds, shuffle = True)
	skf = StratifiedKFold(n_splits = numfolds, shuffle = True)

	for i in range(1, len_class+1):
		for j in range(1,len_phase+1):
			print("**************mode #", i, "****phase", j)

			for batch, label, dtype in tqdm(wholeloaders[k], disable=args.progressbar):
				X = batch
				y = label
				types = dtype


			scale = preprocessing.StandardScaler()
			pca = PCA()
			scale_PCA = Pipeline([('norm',scale),('dimred',pca)])
		
			# print("TRAIN:", len(train_index), "TEST:", len(test_index), 'percentage', len(test_index)/len(train_index))

			m = 0
			for train_index, test_index in skf.split(X, y, types):
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]
				types_train, types_test = types[train_index], types[test_index]
				
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

				# ss_accuracies[i-1][j-1].append(ss_acc) if tot_steady_state != 0 else "No steady state samples used"
				# tr_accuracies[i-1][j-1].append(tr_acc) if tot_transitional != 0 else "No transitional samples used"
				# accuracies[i-1][j-1].append(accuracy_score(y_test, y_pred))

				# tot_corrects[i-1][j-1].append(correct)
				# steady_state_corrects[i-1][j-1].append(steady_state_correct) if tot_steady_state != 0 else "No steady state samples used"
				# transitional_corrects[i-1][j-1].append(transitional_correct) if tot_transitional != 0 else "No transitional samples used"


				# print(accuracy_score(y_test, y_pred))
				# print("Total accuracy: {}".format(accuracy_score(y_test, y_pred)))
				# print("Total correct: {}, number: {}, accuracy: {}".format(correct,tot,tot_acc))
				# print("Steady-state correct: {}, number: {}, accuracy: {}".format(steady_state_correct,tot_steady_state,ss_acc))
				# print("Transistional correct: {}, number: {}, accuracy: {}".format(transitional_correct,tot_transitional,tr_acc))
				# print(accuracy_score(y_test, y_pred))		
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
	print('total number of data: ' ,BIO_trains_len)
	# print('Accuracy_total:', correct/BIO_trains_len)
	# print('Steady-state:', steady_state_correct/tot_steady_state )
	# print('Transistional:', transitional_correct/tot_transitional)

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
import argparse

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
