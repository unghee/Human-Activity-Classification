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

## for N way classifieres
numb_class1 = 6
numb_class2 = 2
numb_class3 = 2
numb_class4 = 2
numb_class5 = 2
numb_class6 = 2

# numb_class = 6

num_epoch = 4

BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


############################################

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# Load the dataset and train, val, test splits
print("Loading datasets...")

## calling for the first time
BIO_train= EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1,48),window_size=500,processed=True)
BIO_val= EnableDataset(subject_list= ['156','185','186','189','190', '191', '192', '193', '194'],data_range=(48,49),window_size=500,processed=True)
BIO_test= EnableDataset(subject_list= ['156','185','189','190', '192', '193', '194'],data_range=(49,50),window_size=500,processed=True)
# BIO_train= EnableDataset(subject_list= ['156','185'],data_range=(1,5),window_size=500,processed=True,transform=transforms.Compose([
#                                                transforms.RandomVerticalFlip(p=0.5),
#                                                transforms.RandomHorizontalFlip()
#                                            ]))
# BIO_val= EnableDataset(subject_list= ['156'],data_range=(48,49),window_size=500,processed=True)
# BIO_test= EnableDataset(subject_list= ['156'],data_range=(49,50),window_size=500,processed=True)



# ## saving dataset a file

save_object(BIO_train, 'BIO_train_.pkl')
save_object(BIO_val, 'BIO_val_.pkl')
save_object(BIO_test, 'BIO_test_.pkl')

# ## load from saved files
# with open('BIO_train_.pkl', 'rb') as input:
#     BIO_train = pickle.load(input)
# with open('BIO_val_.pkl', 'rb') as input:
#     BIO_val = pickle.load(input)
# with open('BIO_test_.pkl', 'rb') as input:
#     BIO_test = pickle.load(input)


## check the class distribution

trainloader = DataLoader(BIO_train, shuffle=False,batch_size=BATCH_SIZE)
classes = [0,0,0,0,0,0,0]
for data, labels in trainloader:
    labels = labels[0]
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
        target = target[0]
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


################# MODEL1#####################
model = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, numb_class1)

model = model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


################# MODEL2#####################
model2 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
num_ftrs = model2.fc.in_features
model2.fc = nn.Linear(num_ftrs, numb_class2)

model2 = model2.to(device)
model2.eval()
criterion2 = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


################# MODEL3#####################
model3 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
num_ftrs = model3.fc.in_features
model3.fc = nn.Linear(num_ftrs, numb_class3)

model3 = model3.to(device)
model3.eval()
criterion3 = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer3 = optim.Adam(model3.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


################# MODEL4#####################
model4 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
num_ftrs = model4.fc.in_features
model4.fc = nn.Linear(num_ftrs, numb_class4)

model4 = model4.to(device)
model4.eval()
criterion4 = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer4 = optim.Adam(model4.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength

################# MODEL5#####################
model5 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
num_ftrs = model5.fc.in_features
model5.fc = nn.Linear(num_ftrs, numb_class5)

model5 = model5.to(device)
model5.eval()
criterion5 = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer5 = optim.Adam(model5.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


################# MODEL6#####################
model6 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
num_ftrs = model6.fc.in_features
model6.fc = nn.Linear(num_ftrs, numb_class6)

model6 = model5.to(device)
model6.eval()
criterion6 = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer6 = optim.Adam(model6.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength



def train(model, loader, num_epoch = 20): # Train the model
    loss_history=[]
    loss_history2=[]
    loss_history3=[]
    loss_history4=[]
    loss_history5=[]
    loss_history6=[]
    val_history=[]
    print("Start training...")
    model.train() # Set the model to training mode
    model2.train()
    model3.train()
    model4.train()
    model5.train()
    model6.train()

    for i in range(num_epoch):
        running_loss = []
        running_loss2 = []
        running_loss3 = []
        running_loss4 = []
        running_loss5 = []
        running_loss6 = []

        for batch, label in tqdm(loader):
            batch = batch.to(device)
            labels = label.to(device)

            label= labels[0,0]
            incom_label = labels[0,1]

            label=label.unsqueeze(0)
            incom_label=incom_label.unsqueeze(0)


            # batch = batch.float()
            # label = label.float()
            # plt.imshow(batch.cpu().detach().numpy().transpose(2,1,0))
            # plt.show()

            ####
            # 1 : LW, 2:RA, 3:RD, 4:SA, 5:SD, 6:Stand
            # MODEL1: LW(1)-> LW(1), RA(2), RD(3), SA(4), SD(5), Stand(6) 
            # MODEL2: RA(2)-> LW(1), RA(2)
            # MODEL3: RD(3)-> LW(1), RD(3)
            # MODEL4: SA(4)-> LW(1), SA(4)
            # MODEL5: SD(5)-> LW(1), SD(5)
            # MODEL6: Stand(6)-> LW(1), SD(6)

            # batch calculation?
            ####
            if incom_label ==1: #LW  

                label = label -1 # indexing start from 1 (removing sitting conditon)
                optimizer.zero_grad() # Clear gradients from the previous iteration
                pred = model(batch) # This will call Network.forward() that you implement
                loss = criterion(pred, label) # Calculate the loss
                running_loss.append(loss.item())
                loss.backward() # Backprop gradients to all tensors in the network
                optimizer.step() # Update trainable weightspre
                loss_history.append(np.mean(running_loss))

            elif incom_label ==2:

                label = label -1 # indexing start from 1 (removing sitting conditon)
                optimizer2.zero_grad() # Clear gradients from the previous iteration
                pred2 = model2(batch) # This will call Network.forward() that you implement
                loss2 = criterion2(pred2, label) # Calculate the loss
                running_loss2.append(loss2.item())
                loss2.backward() # Backprop gradients to all tensors in the network
                optimizer2.step() # Update trainable weightspre
                loss_history2.append(np.mean(running_loss2))

            elif incom_label ==3:
                if label ==3:
                    label = label -2 # when label is 3
                else:
                    label = label -1 # when label is 1

                optimizer3.zero_grad() # Clear gradients from the previous iteration
                pred3 = model3(batch) # This will call Network.forward() that you implement
                loss3 = criterion3(pred3, label) # Calculate the loss
                running_loss3.append(loss3.item())
                loss3.backward() # Backprop gradients to all tensors in the network
                optimizer3.step() # Update trainable weightspre
                loss_history3.append(np.mean(running_loss3))

            elif incom_label ==4:

                if label ==4:
                    label = label -3 # when label is 4
                else:
                    label = label -1 # when label is 1

                optimizer4.zero_grad() # Clear gradients from the previous iteration
                pred4 = model4(batch) # This will call Network.forward() that you implement
                loss4 = criterion4(pred4, label) # Calculate the loss
                running_loss4.append(loss4.item())
                loss4.backward() # Backprop gradients to all tensors in the network
                optimizer4.step() # Update trainable weightspre
                loss_history4.append(np.mean(running_loss4))

            elif incom_label ==5:

                if label ==5:
                    label = label -4 # when label is 5
                else:
                    label = label -1 # when label is 1

                optimizer5.zero_grad() # Clear gradients from the previous iteration
                pred5 = model5(batch) # This will call Network.forward() that you implement
                loss5 = criterion5(pred5, label) # Calculate the loss
                running_loss5.append(loss5.item())
                loss5.backward() # Backprop gradients to all tensors in the network
                optimizer5.step() # Update trainable weightspre
                loss_history5.append(np.mean(running_loss5))

            elif incom_label ==6:

                if label ==6:
                    label = label -5 # when label is 5
                else:
                    label = label -1 # when label is 1

                optimizer6.zero_grad() # Clear gradients from the previous iteration
                pred6 = model6(batch) # This will call Network.forward() that you implement
                loss6 = criterion6(pred6, label) # Calculate the loss
                running_loss6.append(loss6.item())
                loss6.backward() # Backprop gradients to all tensors in the network
                optimizer6.step() # Update trainable weightspre
                loss_history6.append(np.mean(running_loss6))


        val_acc = evaluate(model,model2,model3,model4,model5,model6, valloader)
        val_history.append(val_acc)
        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc)) # Print the average loss for this epoch
    print("Done!")
    return loss_history, val_history

def evaluate(model,model2,model3 ,model4,model5,model6,loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    model2.eval() # Set the model to evaluation mode
    model3.eval() # Set the model to evaluation mode
    model4.eval() # Set the model to evaluation mode
    model5.eval() # Set the model to evaluation mode
    model6.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            labels = label.to(device) 
            label = labels[0,0]
            incom_label = labels[0,1]          
            if incom_label ==1:
                pred = model(batch)
            elif incom_label ==2:
                pass 
                pred =model2(batch)
                # pred_list=pred.tolist()
                pred_list = [[float(pred[0,0]),float(pred[0,1]),-10.0**20,-10.0**20,-10.0**20,-10.0**20]]
                pred = torch.LongTensor(np.array(pred_list))  
                             
            elif incom_label ==3:
                # pass 
                pred =model3(batch)
                # pred_list=pred.tolist()
                pred_list = [[float(pred[0,0]),-10.0**20,float(pred[0,1]),-10.0**20,-10.0**20,10.0**20]]
                pred = torch.LongTensor(np.array(pred_list))
            elif incom_label==4:
                # pass 
                pred =model4(batch)
                # pred_list=pred.tolist()
                pred_list = [[float(pred[0,0]),-10.0**20,-10.0**20,float(pred[0,1]),-10.0**20,-10.0**20]]
                pred = torch.LongTensor(np.array(pred_list))
            elif incom_label ==5:
                # pass 
                pred =model5(batch)
                # pred_list=pred.tolist()
                pred_list = [[float(pred[0,0]),-10.0**20,-10.0**20,-10.0**20,float(pred[0,1]),-10.0**20]]
                pred = torch.LongTensor(np.array(pred_list))
            elif incom_label ==6:
                # pass 
                pred =model6(batch)
                # pred_list=pred.tolist()
                pred_list = [[float(pred[0,0]),-10.0**20,-10.0**20,-10.0**20,-10.0**20,float(pred[0,1])]]
                pred = torch.LongTensor(np.array(pred_list))


            # del pred_list


            label = label-1
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc

loss_history, val_history =train(model, trainloader, num_epoch)
# print("Evaluate on validation set...")
# evaluate(model,model2, valloader)
print("Evaluate on test set")
evaluate(model,model2,model3,model4,model5,model6, testloader)

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



