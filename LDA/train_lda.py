import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm # Displays a progress bar


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataset import EnableDataset

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold


# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=16,hop_length=27)
BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'])

# train_size = int(0.8 * len(BIO_train))
# test_size = int((len(BIO_train) - train_size)/2)
# train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(BIO_train, [train_size, test_size, test_size])


# trainloader = DataLoader(train_dataset, batch_size=len(train_dataset))
# valloader = DataLoader(test_dataset, batch_size=len(val_dataset))
# testloader = DataLoader(val_dataset, batch_size=len(test_dataset))

wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))

# print('train size',len(train_dataset))
# print('val size',len(val_dataset))
# print('test size',len(test_dataset))
# device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
# print('GPU USED?',torch.cuda.is_available())

# run lda without mode specific
# run lda with mode specific

model = LinearDiscriminantAnalysis()

# Define cross-validation parameters
numfolds = 10
# kf = KFold(n_splits = numfolds, shuffle = True)
skf = StratifiedKFold(n_splits = numfolds, shuffle = True)



for batch, label in tqdm(wholeloader):
	X = batch
	y = label 
accuracies =[]
for train_index, test_index in skf.split(X, y):
	# print("TRAIN:", len(train_index), "TEST:", len(test_index), 'percentage', len(test_index)/len(train_index))

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	accuracy_cur=accuracy_score(y_test, y_pred)
	accuracies.append(accuracy_cur)
	print('Accuracy' + str(accuracy_cur))

print('Accuracy_total:', np.mean(accuracies))
# for batch, label in tqdm(trainloader):
# 	model.fit(batch, label)

# for batch, label in tqdm(valloader):
# 	model.score(batch,label)
# 	y_pred = model.predict(batch)
# 	y_test=label


# print('Accuracy' + str(accuracy_score(y_test, y_pred)))