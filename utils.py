import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset

from dataset import EnableDataset

import pickle

from sklearn.model_selection import KFold, StratifiedKFold,ShuffleSplit ,train_test_split

import copy
import os
import random


class trainclass():
	def __init__(self,model,optimizer,databool,device,criterion,model_name):
		self.model = model
		self.optimizer = optimizer
		self.data_bool = databool
		self.device = device
		self.criterion=criterion
		self.model_name= model_name


	def train(self, loader,num_epoch = 20): # Train the model
	    loss_history=[]
	    val_history=[]
	    print("Start training...")
	    self.model.train()
	    pre_loss=10000
	    for i in range(num_epoch):
	        running_loss = []
	        for batch, label in tqdm(loader,disable=self.data_bool):
	            batch = batch.to(self.device)
	            label = label.to(self.device)
	            label = label -1 # indexing start from 1 (removing sitting conditon)
	            self.optimizer.zero_grad()
	            pred = self.model(batch)
	            loss = self.criterion(pred, label)
	            running_loss.append(loss.item())
	            loss.backward()
	            self.optimizer.step()
	            
	        # val_acc = evaluate(model, valloader)
	        # val_history.append(val_acc)
	        loss_mean= np.mean(running_loss)
	        loss_history.append(loss_mean)
	        val_acc =0
	        if loss_mean< pre_loss:
	        	# print(loss_mean,pre_loss)
	        	pre_loss = loss_mean
	        	torch.save(self.model.state_dict(), self.model_name)
	        	print("*model saved*")
	        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc))
	    print("Done!")
	    return loss_history, val_history



	def evaluate(self,loader):
	    self.model.eval()
	    correct = 0
	    with torch.no_grad():
	        count = 0
	        totalloss = 0
	        for batch, label in tqdm(loader,disable=self.data_bool):
	            batch = batch.to(self.device)
	            label = label-1 # indexing start from 1 (removing sitting conditon)
	            label = label.to(self.device)
	            pred = self.model(batch)
	            totalloss += self.criterion(pred, label)
	            count +=1
	            correct += (torch.argmax(pred,dim=1)==label).sum().item()
	    acc = correct/len(loader.dataset)

	    print("Evaluation loss: {}".format(totalloss/count))
	    print("Evaluation accuracy: {}".format(acc))
	    return acc




def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def weight_classes(dataset):
    trainloader = DataLoader(dataset, shuffle=False,batch_size=BATCH_SIZE)
    classes = [0,0,0,0,0,0,0]
    for data, labels in trainloader:
        for x in range(labels.size()[0]):
            classes[labels[x]] +=1
    print(classes)

    classes= classes[1:-1]


    ## with sample
    weights=[]
    sum_classes = np.sum(classes)
    for idx in classes:
        if idx != 0 :
            weights.append(sum_classes/idx)
        else:
            continue

    print(weights)
    weights = torch.FloatTensor(weights)


    return weights

