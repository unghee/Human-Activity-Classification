import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm # Displays a progress bar


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataset import EnableDataset

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold


## for N way classifieres
numb_class= [5,2,2,2,2]
len_class = len(numb_class)
BIO_trains=[]

# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=16,hop_length=27)
# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'])
BIO_trains_len=0
for i in range(len_class):
	BIO_trains.append(EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],prevlabel=i+1))
	BIO_trains_len += len(BIO_trains[i])

# wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))
wholeloaders = []
for i in range(len_class):
	wholeloaders.append(DataLoader(BIO_trains[i],batch_size=len(BIO_trains[i])))





# run lda without mode specific
# run lda with mode specific
models=[]
correct=0
for i in range(len_class):
	model = LinearDiscriminantAnalysis()
	models.append(model)


for i in range(len_class):
	print("**************mode #", i+1)

	# Define cross-validation parameters
	numfolds = 10
	# kf = KFold(n_splits = numfolds, shuffle = True)
	skf = StratifiedKFold(n_splits = numfolds, shuffle = True)


	for batch, label in tqdm(wholeloaders[i]):
		X = batch
		y = label 
	accuracies =[]
	for train_index, test_index in skf.split(X, y):
		# print("TRAIN:", len(train_index), "TEST:", len(test_index), 'percentage', len(test_index)/len(train_index))

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		models[i].fit(X_train, y_train)
		y_pred = models[i].predict(X_test)
		correct += (y_pred==np.array(y_test)).sum().item()
		# accuracy_cur=accuracy_score(y_test, y_pred)
		# accuracies.append(accuracy_cur)
	# print('Accuracy' + str(accuracy_cur))

# print('Accuracy_total:', np.mean(accuracies))

print('Accuracy_total:', correct/BIO_trains_len)

# for batch, label in tqdm(trainloader):
# 	model.fit(batch, label)

# for batch, label in tqdm(valloader):
# 	model.score(batch,label)
# 	y_pred = model.predict(batch)
# 	y_test=label


# print('Accuracy' + str(accuracy_score(y_test, y_pred)))