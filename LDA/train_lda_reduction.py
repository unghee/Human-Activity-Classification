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


# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=16,hop_length=27)
BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'])

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


for batch, label, dtype in tqdm(wholeloader):
	X = batch
	y = label
	types = dtype

model = LinearDiscriminantAnalysis()


X_train, X_test, y_train, y_test = train_test_split((X,types), y, test_size=0.3, random_state=1)
X_train, types_train = X_train
X_test, types_test = X_test

accuracies =[]
i = 0
for train_index, val_index in skf.split(X_train, y_train, types_tain):

	X_train, X_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]
	types_train, types_val = types[train_index], types[val_index]

	model.fit(X_train, y_train)
	y_pred = model.predict(X_val)
	correct += (y_pred==np.array(y_val)).sum().item()
	steady_state_correct += (np.logical_and(y_pred==np.array(y_val), types_val == 1)).sum().item()
	tot_steady_state += (types == 1).sum().item()
	transitional_correct += (np.logical_and(y_pred==np.array(y_val), types_val == 0)).sum().item()
	tot_transitional += (types == 0).sum().item()
	accuracies.append(accuracy_score(y_val, y_pred))
	i +=1


	print(accuracy_score(y_val, y_pred))



# # print('Accuracy_total:', correct/len(BIO_train))
# print('Accuracy_,mean:', np.mean(accuracies),'Accuracy_std: ', np.std(accuracies))
# model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))


### W/ dimension reduction
scale = preprocessing.StandardScaler()
pca = PCA()
scale_PCA = Pipeline([('norm',scale),('dimred',pca)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# scores = cross_val_score(pipe, X_train, y_train, cv=5)
# scores = cross_val_score(model, X_train, y_train, cv=5)
# print('Validation accuracy: %.3f %s' % ( scores.mean(), scores))
accuracies =[]
i = 0
for train_index, val_index in skf.split(X_train, y_train):

	X_train, X_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]

	scale.fit(X_train)
	scale_PCA.fit(X_train)

	feats_train_PCA = scale_PCA.transform(X_train)
	feats_test_PCA = scale_PCA.transform(X_test)

	pcaexplainedvar = np.cumsum(scale_PCA.named_steps['dimred'].explained_variance_ratio_)
	pcanumcomps = min(min(np.where(pcaexplainedvar > 0.95))) + 1

	unique_modes = np.unique(y_train)
	model.set_params(priors = np.ones(len(unique_modes))/len(unique_modes))

	pcaldafit = model.fit(feats_train_PCA[:,0:pcanumcomps],y_train)
	y_pred=pcaldafit.predict(feats_test_PCA[:,0:pcanumcomps]).ravel()
	correct += (y_pred==np.array(y_test)).sum().item()

	accuracies.append(accuracy_score(y_val, y_pred))

	print(accuracy_score(y_val, y_pred))

	i +=1
