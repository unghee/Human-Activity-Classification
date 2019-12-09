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
        self.fc2 = nn.Linear(2000, 7)


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
# Load the dataset and train, val, test splits
print("Loading datasets...")


#BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1,40),processed=True)
#BIO_val= EnableDataset(subject_list= ['156','185','186','189','190', '191', '192', '193', '194'],data_range=(40,45),processed=True)
#BIO_test= EnableDataset(subject_list= ['156','185','189','190', '192', '193', '194'],data_range=(45,50),processed=True)

BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(50, 50, 45, 40, 50, 50, 45, 50, 50, 50),processed=True)
#BIO_train= EnableDataset(subject_list= ['156','185'],data_range=(15, 15),processed=True)
#train_dataset= EnableDataset(data_range=(5,5),processed=True)
#test_dataset= EnableDataset(data_range=(5,5),processed=True)
#val_dataset= EnableDataset(data_range=(5,5),processed=True)

train_size = int(0.8 * len(BIO_train))
test_size = int((len(BIO_train) - train_size)/2)
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(BIO_train, [train_size, test_size, test_size])
# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
classes = [0,0,0,0,0,0,0]
for data, labels in trainloader:
    for x in range(labels.size()[0]):
        classes[labels[x]] +=1
print(classes)
valloader = DataLoader(test_dataset, batch_size=32)
testloader = DataLoader(val_dataset, batch_size=32)

numb_class = 7


device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())
# model = Network().to(device)

#model = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', num_classes=numb_class) # use resnet
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, numb_class)
model = Network()

model = model.to(device)
#model.eval()
weights = torch.FloatTensor([0.0, 1.0, 9693/2609, 9693/3250, 9693/1181, 9693/1133, 9693/530 ])
weights = weights.to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
num_epoch = 60


# TODO: Choose an appropriate number of training epochs

def train(model, loader, num_epoch = 20): # Train the model
    loss_history=[]
    val_history=[]
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            # batch = batch.float()
            # label = label.float()
            # plt.imshow(batch.cpu().detach().numpy().transpose(2,1,0))
            # plt.show()

            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weightspre
            loss_history.append(np.mean(running_loss))
        val_acc = evaluate(model, valloader)
        val_history.append(val_acc)
        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc)) # Print the average loss for this epoch
    print("Done!")
    return loss_history, val_history

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        count = 0
        totalloss = 0
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            totalloss += criterion(pred, label)
            count +=1
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation loss: {}".format(totalloss/count))
    print("Evaluation accuracy: {}".format(acc))
    return acc

loss_history, val_history =train(model, trainloader, num_epoch)
print("Evaluate on validation set...")
evaluate(model, valloader)
print("Evaluate on test set")
evaluate(model, testloader)



