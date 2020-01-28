import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm # Displays a progress bar


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataset import EnableDataset

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA, sparse_encode
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

RESULT_NAME= './results/LDA/accuracy_nway.txt'



## for N way classifieres
numb_class= [5,2,2,2,2]
len_class = len(numb_class)
len_phase = 4
BIO_trains=[]

BIO_trains_len=0

k = 0 
for i in range(1,len_class+1):
	for j in range(1,len_phase+1):
		BIO_trains.append(EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],phaselabel=j,prevlabel=i))
		BIO_trains_len += len(BIO_trains[k])
		# print(k)
		k +=1


wholeloaders = []

k = 0 
for i in range(1,len_class+1):
	for j in range(1,len_phase+1):
		wholeloaders.append(DataLoader(BIO_trains[k],batch_size=len(BIO_trains[k])))
		# print(k)
		k +=1

models=[]
correct=0
for i in range(1,len_class+1):
	for j in range(1,len_phase+1):
		model = LinearDiscriminantAnalysis()
		models.append(model)


k =0 
for i in range(1, len_class+1):
	for j in range(1,len_phase+1):
		print("**************mode #", i, "****phase", j)

		# Define cross-validation parameters
		numfolds = 10
		# kf = KFold(n_splits = numfolds, shuffle = True)
		skf = StratifiedKFold(n_splits = numfolds, shuffle = True)


		for batch, label in tqdm(wholeloaders[k]):
			X = batch
			y = label 

		# pca = PCA()
		# pca_object = pca.fit(X)
		# X_pca = pca_object.transform(X)

		scale = preprocessing.StandardScaler()
		pca = PCA()
		scale_PCA = Pipeline([('norm',scale),('dimred',pca)])



		accuracies=[[[] for x in range(len_class) ]for y in range(len_phase)]

		for train_index, test_index in skf.split(X, y):
			# print("TRAIN:", len(train_index), "TEST:", len(test_index), 'percentage', len(test_index)/len(train_index))

			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			# models[k].fit(X_train, y_train)
			# y_pred = models[k].predict(X_test)

			## dimension reduction
			scale.fit(X_train)
			scale_PCA.fit(X_train)

			feats_train_PCA = scale_PCA.transform(X_train)
			feats_test_PCA = scale_PCA.transform(X_test)   

			pcaexplainedvar = np.cumsum(scale_PCA.named_steps['dimred'].explained_variance_ratio_)                
			pcanumcomps = min(min(np.where(pcaexplainedvar > 0.95))) + 1

			unique_modes = np.unique(y_train)
			models[k].set_params(priors = np.ones(len(unique_modes))/len(unique_modes))

			pcaldafit = models[k].fit(feats_train_PCA[:,0:pcanumcomps],y_train)
			y_pred=pcaldafit.predict(feats_test_PCA[:,0:pcanumcomps]).ravel()

			correct += (y_pred==np.array(y_test)).sum().item()
			print(accuracy_score(y_test, y_pred))

			accuracies[i,j].append(accuracy_score(y_test, y_pred))




		k +=1
	# del pca, pca_object,X_pca
	# del pca, pcanumcomps, X_train, X_test, y_train, y_test

print('total number of classifiers: ' ,k)
print('Accuracy_total:', correct/BIO_trains_len)

print('writing...')
with open(RESULT_NAME, 'w') as f:
	for item in accuracies:
		f.write("%s\n" % item)
f.close()