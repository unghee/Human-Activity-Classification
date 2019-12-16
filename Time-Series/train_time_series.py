import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms , utils
from torch.utils.data import Dataset, Subset, DataLoader, random_split

from dataset import EnableDataset

from tempfile import TemporaryFile
import pickle
from skimage import io, transform


########## SETTINGS  ########################

numb_class = 7
num_epoch = 10

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


############################################

# Load data and split into training (80%), test (10%) and validation (10%)
BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1,50), time_series=True)
train_size = int(0.8 * len(BIO_train))
test_size = int((len(BIO_train) - train_size)/2)
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(BIO_train, [train_size, test_size, test_size])

trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
classes = [0,0,0,0,0,0,0]
for data, labels in trainloader:
    for x in range(labels.size()[0]):
        classes[labels[x]] +=1

valloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
testloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.sclayer1 = nn.Sequential(
            nn.Conv1d(52, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.sclayer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
            )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(32000, 2000)
        self.fc2 = nn.Linear(2000, numb_class)

    def forward(self,x):
        x = self.sclayer1(x)
        x = self.sclayer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print('GPU USED?',torch.cuda.is_available())


model = Network().to(device)
model.eval()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def train(model, loader, num_epoch = 20):
    loss_history=[]
    val_history=[]
    print("Start training...")
    model.train()

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
        val_history.append(val_acc)
        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc))
    print("Done!")
    return loss_history, val_history

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval()
    correct = 0
    correctnum = [0,0,0,0,0,0,0]
    totalnum = [0,0,0,0,0,0,0]
    with torch.no_grad():
        count = 0
        totalloss = 0
        for batch, label in tqdm(loader):
            ct = label.numpy()
            for i in range(len(correctnum)):
                totalnum[i] += np.sum(ct==i)
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            totalloss += criterion(pred, label)
            count +=1
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
            for i in range(len(correctnum)):
                outs = torch.argmax(pred,dim=1).cpu().numpy()
                correctnum[i] += np.sum(np.logical_and(outs==i, outs==ct))
    acc = correct/len(loader.dataset)
    print("Evaluation loss: {}".format(totalloss/count))
    print("Evaluation accuracy: {}".format(acc))
    for i in range(len(correctnum)):
        print(correctnum[i]/totalnum[i])
    return acc

# Train
loss_history, val_history = train(model, trainloader, num_epoch)

# Evaluate on validation and test
print("Evaluate on validation set...")
evaluate(model, valloader)
print("Evaluate on test set")
evaluate(model, testloader)


# Save data about model
np.savetxt('loss_history.txt',loss_history)
np.savetxt('val_history.txt',val_history)

fig =plt.figure()
plt.plot(loss_history,label='train loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.show()
fig.savefig('train_loss.jpg')

fig =plt.figure()
plt.plot(val_history,label='vaidation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
list_idx = [ i for i in range(num_epoch)]
plt.xticks(np.array(list_idx))
plt.legend()
plt.show()
fig.savefig('val_acc.jpg')
torch.save(model.state_dict(), 'time_series_model.pth')



