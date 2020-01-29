import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm # Displays a progress bar


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataset import EnableDataset

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.decomposition import PCA, sparse_encode
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os



MODE = 'bilateral'
CLASSIFIER = 'LDA'


RESULT_NAME= './results/LDA/'+CLASSIFIER+'_accuracy_'+MODE+'.txt'


if not os.path.exists('./results/LDA'):
	os.makedirs('./results/LDA')

# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=16,hop_length=27)
BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'])
# BIO_train= EnableDataset(subject_list= ['156'])
wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))

correct=0


# Define cross-validation parameters
numfolds = 10
skf = KFold(n_splits = numfolds, shuffle = True)
# skf = StratifiedKFold(n_splits = numfolds, shuffle = True)


scale = preprocessing.StandardScaler()
pca = PCA()
scale_PCA = Pipeline([('norm',scale),('dimred',pca)])

for batch, label in tqdm(wholeloader):
	X = batch
	y = label 

if CLASSIFIER == 'LDA':
	model = LinearDiscriminantAnalysis()
elif CLASSIFIER == 'SVM':
	model = SVC(kernel = 'linear', C = 10)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
accuracies =[]
i = 0
for train_index, test_index in skf.split(X, y):
	
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	
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


	correct += (y_pred==np.array(y_test)).sum().item()
	accuracies.append(accuracy_score(y_test, y_pred))
	i +=1

	print(accuracy_score(y_test, y_pred))



print('Accuracy_total:', correct/len(BIO_train))
print('Accuracy_,mean:', np.mean(accuracies),'Accuracy_std: ', np.std(accuracies))
# model.fit(X_train, y_train)

print('writing...')
with open(RESULT_NAME, 'w') as f:
	for item in accuracies:
		f.write("%s\n" % item)
f.close()