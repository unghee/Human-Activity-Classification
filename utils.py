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
import pdb

class trainclass():
	def __init__(self,model,optimizer,databool,device,criterion,model_name,batch_size):
		self.model = model
		self.optimizer = optimizer
		self.data_bool = databool
		self.device = device
		self.criterion=criterion
		self.model_name= model_name
		self.batch_size = batch_size


	def train(self, loader,num_epoch = 20,onehot=None): # Train the model
	    loss_history=[]
	    val_history=[]
	    print("Start training...")
	    self.model.train()
	    pre_loss=10000
	    for i in range(num_epoch):
	        running_loss = []
	        for batch, label, types in tqdm(loader,disable=self.data_bool):
	            batch = batch.to(self.device)
	            label = label.to(self.device)
	            label = label -1 # indexing start from 1 (removing sitting conditon)
	            self.optimizer.zero_grad()
	            pred = self.model(batch)
	            loss = self.criterion(pred, label)
	            running_loss.append(loss.item())
	            loss.backward()
	            self.optimizer.step()

	        loss_mean= np.mean(running_loss)
	        loss_history.append(loss_mean)
	        val_acc =0
	        if loss_mean< pre_loss:
	        	pre_loss = loss_mean
	        	torch.save(self.model.state_dict(), self.model_name)
	        	print("*model saved*")
	        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc))
	    print("Done!")
	    return loss_history, val_history

	def train_modesp(self, loader,num_epoch = 20): # Train the model
	    loss_history=[]
	    val_history=[]
	    print("Start training...")
	    self.model.train()
	    pre_loss=10000
	    for i in range(num_epoch):
	        running_loss = []
	        for batch, label, types, onehot in tqdm(loader,disable=self.data_bool):
	            batch = batch.to(self.device)
	            label = label.to(self.device)
	            onehot = onehot.to(self.device)
	            label = label -1 # indexing start from 1 (removing sitting conditon)
	            self.optimizer.zero_grad()
	            pred = self.model(batch,onehot)
	            # pred = self.model([batch,onehot]) # For mode-specific resnet
	            loss = self.criterion(pred, label)
	            running_loss.append(loss.item())
	            loss.backward()
	            self.optimizer.step()

	        loss_mean= np.mean(running_loss)
	        loss_history.append(loss_mean)
	        val_acc =0
	        if loss_mean< pre_loss:
	        	pre_loss = loss_mean
	        	torch.save(self.model.state_dict(), self.model_name)
	        	print("*model saved*")
	        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc))
	    print("Done!")
	    return loss_history, val_history

	def train_LSTM(self,loader,valloader=None, num_epoch =20,onehot=None): # Train the model\
		loss_history=[]
		val_history=[]
		print("Start training...")

		pre_loss=10000
		for i in range(num_epoch):
			self.model.train()
			running_loss = []
			h = self.model.init_hidden(self.batch_size)  
			for batch, label, types in tqdm(loader,disable=self.data_bool):
				batch = batch.to(self.device)
				label = label.to(self.device)
				label = label -1 # indexing start from 1 (removing sitting conditon)
				h = tuple([e.data for e in h])
				# h = tuple([each.repeat(1, BATCH_SIZE, 1).data for each in h])
				self.optimizer.zero_grad()
				pred,h = self.model(batch,h,self.batch_size)
				loss = self.criterion(pred, label.long())
				running_loss.append(loss.item())
				loss.backward()
				self.optimizer.step()
				loss_mean= np.mean(running_loss)
			loss_history.append(loss_mean)
			# val_acc = 0
			val_acc,_,_,_,_,_=self.evaluate_LSTM(valloader,False)
			if loss_mean< pre_loss:
				pre_loss = loss_mean
				torch.save(self.model.state_dict(), self.model_name)
				print("*model saved*")
			print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc))
			# print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss)))
			print("Done!")
		return loss_history, val_history


	def evaluate(self,loader):
	    self.model.eval()
	    correct = 0
	    steady_state_correct = 0
	    tot_steady_state = 0
	    transitional_correct = 0
	    tot_transitional = 0
	    preds=[]
	    tests=[]

	    class_correct = [0]*6
	    class_total = [0]*6
	    class_acc=[]

	    with torch.no_grad():
	        count = 0
	        totalloss = 0
	        for batch, label, types in tqdm(loader,disable=self.data_bool):
	            batch = batch.to(self.device)
	            label = label-1 # indexing start from 1 (removing sitting conditon)
	            label = label.to(self.device)
	            pred = self.model(batch)
	            totalloss += self.criterion(pred, label)
	            count +=1
	            preds.extend((torch.argmax(pred,dim=1)).tolist())
	            tests.extend(label.tolist())

	            correct += (torch.argmax(pred,dim=1)==label).sum().item()
	            steady_state_correct += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), types == 1)).sum().item()
	            tot_steady_state += (types == 1).sum().item()
	            transitional_correct += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), types == 0)).sum().item()
	            tot_transitional += (types == 0).sum().item()

	            for i in range(len(class_correct)):
	                class_correct[i] += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), label.cpu() == i)).sum().item()
	                class_total[i] += (label == i).sum().item()
	    acc = correct/len(loader.dataset)
	    for i in range(len(class_correct)):
	    	if class_total[i] == 0:
	    		print("Class {} has no samples".format(i))
	    	else:
	    		print("Class {} accuracy: {}".format(i, class_correct[i]/class_total[i]))
	    		class_acc.append(class_correct[i]/class_total[i])
	    ss_acc = steady_state_correct/tot_steady_state if tot_steady_state != 0 else "No steady state samples used"
	    tr_acc = transitional_correct/tot_transitional if tot_transitional != 0 else "No transitional samples used"
	    print("Evaluation loss: {}".format(totalloss/count))
	    print("Evaluation accuracy: {}".format(acc))
	    print("Steady-state accuracy: {}".format(ss_acc))
	    print("Transistional accuracy: {}".format(tr_acc))

	    return acc, ss_acc, tr_acc, preds, tests, class_acc


	def evaluate_modesp(self,loader):
	    self.model.eval()
	    correct = 0
	    steady_state_correct = 0
	    tot_steady_state = 0
	    transitional_correct = 0
	    tot_transitional = 0
	    with torch.no_grad():
	        count = 0
	        totalloss = 0
	        for batch, label, types, onehot in tqdm(loader,disable=self.data_bool):
	            batch = batch.to(self.device)
	            label = label-1 # indexing start from 1 (removing sitting conditon)
	            label = label.to(self.device)
	            onehot = onehot.to(self.device)
	            pred = self.model(batch,onehot)
	            totalloss += self.criterion(pred, label)
	            count +=1
	            correct += (torch.argmax(pred,dim=1)==label).sum().item()
	            steady_state_correct += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), types == 1)).sum().item()
	            tot_steady_state += (types == 1).sum().item()
	            transitional_correct += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), types == 0)).sum().item()
	            tot_transitional += (types == 0).sum().item()
	    acc = correct/len(loader.dataset)

	    ss_acc = steady_state_correct/tot_steady_state if tot_steady_state != 0 else "No steady state samples used"
	    tr_acc = transitional_correct/tot_transitional if tot_transitional != 0 else "No transitional samples used"
	    print("Evaluation loss: {}".format(totalloss/count))
	    print("Evaluation accuracy: {}".format(acc))
	    print("Steady-state accuracy: {}".format(ss_acc))
	    print("Transistional accuracy: {}".format(tr_acc))
	    return acc, ss_acc, tr_acc


	def evaluate_LSTM(self, loader,printbool=None):
	    self.model.eval()
	    correct = 0
	    steady_state_correct = 0
	    tot_steady_state = 0
	    transitional_correct = 0
	    tot_transitional = 0
	    preds=[]
	    tests=[]

	    class_correct = [0]*6
	    class_total = [0]*6
	    class_acc=[]

	    h = self.model.init_hidden(self.batch_size) 

	    with torch.no_grad():
	        count = 0
	        totalloss = 0
	        for batch, label, types in tqdm(loader,disable=self.data_bool):
	            batch = batch.to(self.device)
	            label = label-1 # indexing start from 1 (removing sitting conditon)
	            label = label.to(self.device)
	            h = tuple([each.data for each in h])

	            pred,h = self.model(batch,h,self.batch_size)
	            totalloss += self.criterion(pred, label.long())
	            count +=1
	            preds.extend((torch.argmax(pred,dim=1)).tolist())
	            tests.extend(label.tolist())

	            correct += (torch.argmax(pred,dim=1)==label).sum().item()
	            steady_state_correct += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), types == 1)).sum().item()
	            tot_steady_state += (types == 1).sum().item()
	            transitional_correct += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), types == 0)).sum().item()
	            tot_transitional += (types == 0).sum().item()

	            for i in range(len(class_correct)):
	                class_correct[i] += (np.logical_and((torch.argmax(pred,dim=1) == label ).cpu(), label.cpu() == i)).sum().item()
	                class_total[i] += (label == i).sum().item()
	    acc = correct/len(loader.dataset)
	    for i in range(len(class_correct)):
	    	if class_total[i] == 0:
	    		if printbool:
	    			print("Class {} has no samples".format(i))
	    	else:
	    		if printbool:
	    			print("Class {} accuracy: {}".format(i, class_correct[i]/class_total[i]))
	    		class_acc.append(class_correct[i]/class_total[i])
	    ss_acc = steady_state_correct/tot_steady_state if tot_steady_state != 0 else "No steady state samples used"
	    tr_acc = transitional_correct/tot_transitional if tot_transitional != 0 else "No transitional samples used"
	    if printbool:
		    print("Evaluation loss: {}".format(totalloss/count))
		    print("Evaluation accuracy: {}".format(acc))
		    print("Steady-state accuracy: {}".format(ss_acc))
		    print("Transistional accuracy: {}".format(tr_acc))

	    return acc, ss_acc, tr_acc, preds, tests, class_acc


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

