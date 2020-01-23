import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm # Displays a progress bar


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataset import EnableDataset

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.decomposition import PCA, sparse_encode
from sklearn.pipeline import Pipeline

## for N way classifieres


# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=16,hop_length=27)
BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'])


wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))




correct=0






# Define cross-validation parameters
numfolds = 10
skf = KFold(n_splits = numfolds, shuffle = True)
# skf = StratifiedKFold(n_splits = numfolds, shuffle = True)


for batch, label in tqdm(wholeloader):
	X = batch
	y = label 

model = LinearDiscriminantAnalysis()




# pca_object = pca.fit(X)
# X_pca = pca_object.transform(X)

# scale = preprocessing.StandardScaler()
# pca = PCA()
# scale_PCA = Pipeline([('norm',scale),('dimred',pca)])



accuracies =[]
for train_index, test_index in skf.split(X, y):

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]


	## Dimension reduction
	# scale.fit(X_train)
	# scale_PCA.fit(X_train)

	# feats_train_PCA = scale_PCA.transform(X_train)
	# feats_test_PCA = scale_PCA.transform(X_test)   

	# pcaexplainedvar = np.cumsum(scale_PCA.named_steps['dimred'].explained_variance_ratio_)                
	# pcanumcomps = min(min(np.where(pcaexplainedvar > 0.95))) + 1

	# unique_modes = np.unique(y_train)
	# model.set_params(priors = np.ones(len(unique_modes))/len(unique_modes))

	# pcaldafit = model.fit(feats_train_PCA[:,0:pcanumcomps],y_train)
	# y_pred=pcaldafit.predict(feats_test_PCA[:,0:pcanumcomps]).ravel()
	# correct += (y_pred==np.array(y_test)).sum().item()

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	correct += (y_pred==np.array(y_test)).sum().item()
	accuracies.append(accuracy_score(y_test, y_pred))
	print(accuracy_score(y_test, y_pred))


print('Accuracy_total:', correct/len(BIO_train))
print('Accuracy_,mean:', np.mean(accuracies),'Accuracy_std: ', np.std(accuracies))
