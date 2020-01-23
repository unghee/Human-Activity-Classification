import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

from dataset import EnableDataset

import pickle




########## SETTINGS  ########################

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
NUMB_CLASS = 36
NUB_EPOCH=80
############################################


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
    classes = [0] * 36
    for data, labels in trainloader:
        for x in range(labels.size()[0]):
            classes[labels[x]] +=1


    ## with sample
    weights=[]
    sum_classes = np.sum(classes)
    for idx in classes:
        if idx != 0 :
            weights.append(sum_classes/idx)
        else:
            weights.append(0)

    print(weights)
    weights = torch.FloatTensor(weights)


    return weights


# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1, 50),bands=16,hop_length=27)
# BIO_train= EnableDataset(subject_list= ['156'],data_range=(1, 10),bands=16,hop_length=27)

# save_object(BIO_train,'BIO_train_melspectro_36output.pkl')

with open('BIO_train_melspectro_36output.pkl', 'rb') as input:
    BIO_train = pickle.load(input)


train_size = int(0.8 * len(BIO_train))+1
test_size = int((len(BIO_train) - train_size)/2)
print(len(BIO_train), train_size, test_size)
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(BIO_train, [train_size, test_size, len(BIO_train) - train_size - test_size])
# Create dataloaders
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# classes = [0,0,0,0,0,0,0]
# for data, labels in trainloader:
#     for x in range(labels.size()[0]):
#         classes[labels[x]] +=1
# print(classes)
valloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
testloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# numb_class = 5


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
    pre_val_acc =0
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            loss_history.append(np.mean(running_loss))
        val_acc = evaluate(model, valloader)

        if val_acc> pre_val_acc:
        	print(val_acc,pre_val_acc)
        	pre_val_acc = val_acc
        	torch.save(model.state_dict(), './36-output/models/bestmodel.pth')
        	print("########model saved##########")


        val_history.append(val_acc)
        
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
            label = label.to(device)
            pred = model(batch)
            totalloss += criterion(pred, label)
            count +=1
            correct += (torch.argmax(pred,dim=1) % 6 == label% 6).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation loss: {}".format(totalloss/count))
    print("Evaluation accuracy: {}".format(acc))
    return acc

loss_history, val_history =train(model, trainloader, num_epoch)
# print("Evaluate on validation set...")
# evaluate(model, valloader)


model = Network()
model = model.to(device)
model.load_state_dict(torch.load('./36-output/models/bestmodel.pth'))
print("Evaluate on test set")
evaluate(model, testloader)
