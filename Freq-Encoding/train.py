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



########## SETTINGS  ########################

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
NUMB_CLASS = 5
NUB_EPOCH=200
############################################

MODEL_NAME = './Freq-Encoding/models/bestmodel'+ \
        		'_BATCH_SIZE'+str(BATCH_SIZE)+'_LR'+str(LEARNING_RATE)+'_WD'+str(WEIGHT_DECAY)+'_EPOCH'+str(NUB_EPOCH)+'.pth'


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.sclayer1 = nn.Sequential(
            nn.Conv2d(51, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.sclayer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear( 4096, 2000)
        # self.fc1 = nn.Linear( 8192, 2000)
        # self.fc1 = nn.Linear( 20480, 2000)
        self.fc2 = nn.Linear(2000, NUMB_CLASS)


    def forward(self,x):
        x = self.sclayer1(x)
        x = self.sclayer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
# Load the dataset and train, val, test splits
print("Loading datasets...")


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




device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())
model = Network()
model = model.to(device)

# weights = weight_classes(BIO_train)

# weights = torch.FloatTensor([0.0, 1.0, 9693/2609, 9693/3250, 9693/1181, 9693/1133, 9693/530 ])
# weights = weights.to(device)
# criterion = nn.CrossEntropyLoss(weight=weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_epoch = NUB_EPOCH

def train(model, loader, num_epoch = 20): # Train the model
    loss_history=[]
    val_history=[]
    print("Start training...")
    model.train()
    pre_loss=10000
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            label = label -1 # indexing start from 1 (removing sitting conditon)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        # val_acc = evaluate(model, valloader)
        # val_history.append(val_acc)
        loss_mean= np.mean(running_loss)
        loss_history.append(loss_mean)
        val_acc =0
        if loss_mean< pre_loss:
        	print(loss_mean,pre_loss)
        	pre_loss = loss_mean
        	# torch.save(model.state_dict(), './Freq-Encoding/models/bestmodel_.pth')
        	torch.save(model.state_dict(), MODEL_NAME)
        	print("########model saved##########")
        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc))
    print("Done!")
    return loss_history, val_history

def evaluate(model, loader):
    model.eval()
    correct = 0
    # labels = [0,0,0,0,0,0,0]
    # totalcount = [0,0,0,0,0,0,0]
    with torch.no_grad():
        count = 0
        totalloss = 0
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label-1 # indexing start from 1 (removing sitting conditon)
            label = label.to(device)
            pred = model(batch)
            totalloss += criterion(pred, label)
            count +=1
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)

    print("Evaluation loss: {}".format(totalloss/count))
    print("Evaluation accuracy: {}".format(acc))
    return acc






# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=16,hop_length=27)

with open('BIO_train_melspectro_500s_bands_16_hop_length_27.pkl', 'rb') as input:
    BIO_train = pickle.load(input)


wholeloader = DataLoader(BIO_train, batch_size=len(BIO_train))


for batch, label in tqdm(wholeloader):
	X = batch
	y = label 

accuracies =[]

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1)
test_dataset = TensorDataset( X_test, y_test)
testloader =  DataLoader(test_dataset, batch_size=BATCH_SIZE)

train_dataset = TensorDataset( X_train, y_train)
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


train(model, trainloader, num_epoch)


model = Network()
model = model.to(device)
model.load_state_dict(torch.load(MODEL_NAME))

print("Evaluate on test set")
evaluate(model, testloader)

