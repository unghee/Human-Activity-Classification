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

numb_class = 6
num_epoch = 1

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


############################################

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# Load the dataset and train, val, test splits
print("Loading datasets...")

## calling for the first time
# BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1,48),window_size=500,processed=True)
# BIO_val= EnableDataset(subject_list= ['156','185','186','189','190', '191', '192', '193', '194'],data_range=(48,49),window_size=500,processed=True)
# BIO_test= EnableDataset(subject_list= ['156','185','189','190', '192', '193', '194'],data_range=(49,50),window_size=500,processed=True)
BIO_train= EnableDataset(subject_list= ['156','185'],data_range=(1,7),window_size=500,processed=True,transform=transforms.Compose([
                                               transforms.RandomVerticalFlip(p=0.5),
                                               transforms.RandomHorizontalFlip()
                                           ]))
BIO_val= EnableDataset(subject_list= ['156'],data_range=(48,49),window_size=500,processed=True)
BIO_test= EnableDataset(subject_list= ['156'],data_range=(49,50),window_size=500,processed=True)


## saving dataset a file
# outfile = TemporaryFile()
# np.savez(outfile, BIO_train=BIO_train, BIO_val=BIO_val,BIO_test=BIO_test)
# npzfile = np.load(outfile)
# BIO_train,BIO_val,BIO_test = npzfile.files
# save_object(BIO_train, 'BIO_train.pkl')
# save_object(BIO_val, 'BIO_val.pkl')
# save_object(BIO_test, 'BIO_test.pkl')

## load from saved files
# with open('BIO_train.pkl', 'rb') as input:
#     BIO_train = pickle.load(input)
# with open('BIO_val.pkl', 'rb') as input:
#     BIO_val = pickle.load(input)
# with open('BIO_test.pkl', 'rb') as input:
#     BIO_test = pickle.load(input)


## check the class distribution

trainloader = DataLoader(BIO_train, shuffle=False,batch_size=BATCH_SIZE)
classes = [0,0,0,0,0,0,0]
for data, labels in trainloader:
    for x in range(labels.size()[0]):
        classes[labels[x]] +=1
        # print(labels)
print(classes)

classes= classes[1:]


## with sample

sum_classes = np.sum(classes)
weights = [sum_classes/idx  for idx in classes]
print(weights)

idx =0
weight_samples = [0] * len(BIO_train)  
for val, target in BIO_train:
        target= target.numpy()
        target = int(target)
        weight_samples[idx] = weights[target-1]  
        idx +=1


weight_samples= torch.DoubleTensor(weight_samples)   
sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_samples, len(weight_samples)) 
# weights = torch.DoubleTensor(weights) 
trainloader = DataLoader(BIO_train, shuffle=False, sampler=sampler,batch_size=BATCH_SIZE)
valloader = DataLoader(BIO_val, shuffle=False,batch_size=BATCH_SIZE)
testloader = DataLoader(BIO_test, shuffle=False,batch_size=BATCH_SIZE)


## with no sampler
# trainloader = DataLoader(BIO_train, batch_size=32, shuffle=True)
# valloader = DataLoader(BIO_val, batch_size=32,shuffle=True)
# testloader = DataLoader(BIO_test, batch_size=32,shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Design your own network, define layers here.
        # Here We provide a sample of two-layer fully-connected network from HW4 Part3.
        # Your solution, however, should contain convolutional layers.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout
        # If you have many layers, consider using nn.Sequential() to simplify your code
        # self.fc1 = nn.Linear(28*28, 8) # from 28x28 input image to hidden layer of size 256
        # self.fc2 = nn.Linear(8,10) # from hidden layer to 10 class scores
        self.sclayer1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.sclayer2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear( 1* 12 * 24, 500)
        self.fc2 = nn.Linear(500, 7)


    def forward(self,x):
        # TODO: Design your own network, implement forward pass here
        # x = x.view(-1,28*28) # Flatten each image in the batch
        # x = self.fc1(x)
        # relu = nn.ReLU() # No need to define self.relu because it contains no parameters
        # x = relu(x)
        # x = self.fc2(x)
        # # The loss layer will be applied outside Network class
        # x = x.view(-1,28,28,1)

        # torch.Size([1, 3, 4, 51])
        x = self.sclayer1(x) #torch.Size([1, 12, 2, 25])
        x = self.sclayer2(x) #torch.Size([1, 24, 1, 12])
        x = x.reshape(x.size(0), -1) 
        # x = x.view(-1,7 * 7 * 32)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())
# model = Network().to(device)

model = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, numb_class)

model = model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


# TODO: Choose an appropriate number of training epochs

def train(model, loader, num_epoch = 20): # Train the model
    loss_history=[]
    val_history=[]
    print("Start training...")
    model.train() # Set the model to training mode
    # model2.train()

    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            # batch = batch.float()
            # label = label.float()
            # plt.imshow(batch.cpu().detach().numpy().transpose(2,1,0))
            # plt.show()

            ####
            # 1 : LW, 
            ####
            # if label ==1: #LW

            label = label -1 # indexing start from 1 (removing sitting conditon)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weightspre
            loss_history.append(np.mean(running_loss))

            # elif label ==2:
        val_acc = evaluate(model, valloader)
        val_history.append(val_acc)
        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc)) # Print the average loss for this epoch
    print("Done!")
    return loss_history, val_history

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            label = label-1
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc

loss_history, val_history =train(model, trainloader, num_epoch)
print("Evaluate on validation set...")
evaluate(model, valloader)
print("Evaluate on test set")
evaluate(model, testloader)

np.savetxt('loss_history.txt',loss_history)
np.savetxt('val_history.txt',val_history)
# np.savetxt('test_accuracy.txt',acc2)


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



