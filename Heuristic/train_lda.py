import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm # Displays a progress bar


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.decomposition import PCA, sparse_encode
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os, sys
sys.path.append('.')

from dataset import EnableDataset

from itertools import combinations

def run_classifier(mode='bilateral',classifier='LDA',sensor=["imu","emg","goin"]):

	MODE = mode
	CLASSIFIER = classifier
	SENSOR = sensor
	sensor_str='_'.join(SENSOR)


	RESULT_NAME= './results/'+CLASSIFIER+'/'+CLASSIFIER+'_'+MODE+'_'+sensor_str+'_accuracy.txt'


	if not os.path.exists('./results/'+CLASSIFIER):
		os.makedirs('./results/'+CLASSIFIER)

	# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=16,hop_length=27)
	BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],model_type=CLASSIFIER,sensors=SENSOR,mode=MODE)
	
	# BIO_train= EnableDataset(subject_list= ['156'])
	wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))

	correct=0
	steady_state_correct = 0
	tot_steady_state = 0
	transitional_correct = 0
	tot_transitional = 0


	# Define cross-validation parameters
	numfolds = 10
	skf = KFold(n_splits = numfolds, shuffle = True)
	# skf = StratifiedKFold(n_splits = numfolds, shuffle = True)



	scale = preprocessing.StandardScaler()
	pca = PCA()
	scale_PCA = Pipeline([('norm',scale),('dimred',pca)])

	for batch, label, dtype in tqdm(wholeloader):
		X = batch
		y = label
		types = dtype

	if CLASSIFIER == 'LDA':
		model = LinearDiscriminantAnalysis()
	elif CLASSIFIER == 'SVM':
		model = SVC(kernel = 'linear', C = 10)

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
	accuracies =[]
	ss_accuracies=[]
	tr_accuracies=[]

	i = 0
	for train_index, test_index in skf.split(X, y, types):

		print("**************FOLD {}*********".format(i+1))

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		types_train, types_test = types[train_index], types[test_index]

		
		# model.fit(X_train, y_train)
		if CLASSIFIER == 'LDA':

			scale_PCA.fit(X_train)
			feats_train_PCA = scale_PCA.transform(X_train)
			feats_test_PCA = scale_PCA.transform(X_test)	

			pcaexplainedvar = np.cumsum(scale_PCA.named_steps['dimred'].explained_variance_ratio_)                
			pcanumcomps = min(min(np.where(pcaexplainedvar > 0.95))) + 1
			unique_modes = np.unique(y_train)

			model.set_params(priors = np.ones(len(unique_modes))/len(unique_modes))
			model.fit(feats_train_PCA, y_train)
			y_pred = model.predict(feats_test_PCA)

		elif CLASSIFIER == 'SVM':

			scale.fit(X_train)
			feats_train_norm = scale.transform(X_train)
			feats_test_norm = scale.transform(X_test )

			model.fit(feats_train_norm, y_train)
			y_pred = model.predict(feats_test_norm)


		# y_pred = model.predict(X_test)
		# y_pred = model.predict(feats_test_norm)


		correct = (y_pred==np.array(y_test)).sum().item()
		tot = len(y_test)
		steady_state_correct = (np.logical_and(y_pred==np.array(y_test), types_test == 1)).sum().item()
		tot_steady_state = (types_test == 1).sum().item()
		transitional_correct = (np.logical_and(y_pred==np.array(y_test), types_test == 0)).sum().item()
		tot_transitional = (types_test == 0).sum().item()

		accuracies.append(accuracy_score(y_test, y_pred))
		

		tot_acc = correct/tot
		ss_acc = steady_state_correct/tot_steady_state if tot_steady_state != 0 else "No steady state samples used"
		tr_acc = transitional_correct/tot_transitional if tot_transitional != 0 else "No transitional samples used"

		ss_accuracies.append(ss_acc) if tot_steady_state != 0 else "No steady state samples used"
		tr_accuracies.append(tr_acc) if tot_transitional != 0 else "No transitional samples used"



		print("Total accuracy: {}".format(accuracy_score(y_test, y_pred)))
		print("Total correct: {}, number: {}, accuracy: {}".format(correct,tot,tot_acc))
		print("Steady-state correct: {}, number: {}, accuracy: {}".format(steady_state_correct,tot_steady_state,ss_acc))
		print("Transistional correct: {}, number: {}, accuracy: {}".format(transitional_correct,tot_transitional,tr_acc))
		print(accuracy_score(y_test, y_pred))

		i +=1
	print('********************SUMMARY*****************************')
	# print('Accuracy_total:', correct/len(BIO_train))
	print('Accuracy_,mean:', np.mean(accuracies),'Accuracy_std: ', np.std(accuracies))
	print('SR Accuracy_,mean:', np.mean(ss_accuracies),'Accuracy_std: ', np.std(ss_accuracies))
	print('TR Accuracy_,mean:', np.mean(tr_accuracies),'Accuracy_std: ', np.std(tr_accuracies))
	# model.fit(X_train, y_train)
	# total_accuracies = accuracies + ss_accuracies + tr_accuracies


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


classifiers=['LDA']
# sensors=["emg"]
sensors=["imu","emg","goin"]
# modes = ['ipsilateral']
modes = ['bilateral']
for classifier in classifiers:
	for i in range(3,4):
		for combo in combinations(sensors,i):
			sensor = [item for item in combo]
			for mode in modes:
				print(classifier, sensor, mode)
				run_classifier(mode=mode,classifier=classifier,sensor=sensor)


