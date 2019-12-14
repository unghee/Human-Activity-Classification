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
import pdb

from networks import Network

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

BIO_trains=[]
BIO_vals=[]
BIO_tests=[]
for i in range(1,7):
    BIO_trains.append(EnableDataset(subject_list= ['156','185'],data_range=(1,8),window_size=500,processed=True,label=i))
    BIO_vals.append(EnableDataset(subject_list= ['156'],data_range=(40,48),window_size=500,processed=True,label=i))
    BIO_tests.append(EnableDataset(subject_list= ['156'],data_range=(42,50),window_size=500,processed=True,label=i))
    

# BIO_train1= EnableDataset(subject_list= ['156','185'],data_range=(1,8),window_size=500,processed=True,label=1)
# BIO_train2= EnableDataset(subject_list= ['156','185'],data_range=(1,8),window_size=500,processed=True, label=2)
# BIO_train3= EnableDataset(subject_list= ['156','185'],data_range=(1,8),window_size=500,processed=True, label=3)
# BIO_train4= EnableDataset(subject_list= ['156','185'],data_range=(1,8),window_size=500,processed=True, label=4)
# BIO_train5= EnableDataset(subject_list= ['156','185'],data_range=(1,8),window_size=500,processed=True, label=5)
# BIO_train6= EnableDataset(subject_list= ['156','185'],data_range=(1,8),window_size=500,processed=True, label=6)

# BIO_val1= EnableDataset(subject_list= ['156'],data_range=(40,48),window_size=500,processed=True,label=1)
# BIO_val2= EnableDataset(subject_list= ['156'],data_range=(40,48),window_size=500,processed=True,label=2)
# BIO_val3= EnableDataset(subject_list= ['156'],data_range=(40,48),window_size=500,processed=True,label=3)
# BIO_val4= EnableDataset(subject_list= ['156'],data_range=(40,48),window_size=500,processed=True,label=4)
# BIO_val5= EnableDataset(subject_list= ['156'],data_range=(40,48),window_size=500,processed=True,label=5)
# BIO_val6= EnableDataset(subject_list= ['156'],data_range=(40,48),window_size=500,processed=True,label=6)


# BIO_test1= EnableDataset(subject_list= ['156'],data_range=(42,50),window_size=500,processed=True,label=1)
# BIO_test2= EnableDataset(subject_list= ['156'],data_range=(42,50),window_size=500,processed=True,label=2)
# BIO_test3= EnableDataset(subject_list= ['156'],data_range=(42,50),window_size=500,processed=True,label=3)
# BIO_test4= EnableDataset(subject_list= ['156'],data_range=(42,50),window_size=500,processed=True,label=4)
# BIO_test5= EnableDataset(subject_list= ['156'],data_range=(42,50),window_size=500,processed=True,label=5)
# BIO_test6= EnableDataset(subject_list= ['156'],data_range=(42,50),window_size=500,processed=True,label=6)


# ## saving dataset a file

save_object(BIO_trains, 'BIO_trains.pkl')
save_object(BIO_vals, 'BIO_vals')
save_object(BIO_tests, 'BIO_tests')

# ## load from saved files
# with open('BIO_train_.pkl', 'rb') as input:
#     BIO_train = pickle.load(input)
# with open('BIO_val_.pkl', 'rb') as input:
#     BIO_val = pickle.load(input)
# with open('BIO_test_.pkl', 'rb') as input:
#     BIO_test = pickle.load(input)


## check the class distribution

def weight_classes(dataset):
    trainloader = DataLoader(dataset, shuffle=False,batch_size=BATCH_SIZE)
    classes = [0,0,0,0,0,0,0]
    for data, labels in trainloader:
        # labels = labels[0]
        for x in range(labels.size()[0]):
            classes[labels[x]] +=1
            # print(labels)
    # print(classes)

    classes= classes[1:]


    ## with sample
    weights=[]
    sum_classes = np.sum(classes)
    for idx in classes:
        if idx != 0 :
            weights.append(idx)
        else:
            continue
            # weights.append(0)
    # weights = [sum_classes/idx  for idx in classes if idx]
    print(weights)

    # idx =0
    # weight_samples = [0] * len(dataset)  
    # for val, target in dataset:
    #         # target= target.numpy()
    #         # target = target[0]
    #         # pdb.set_trace()
    #         target = int(target)
    #         # pdb.set_trace()
    #         weight_samples[idx] = weights[target-1]  
    #         idx +=1
    # print('weigted_samples',len(weighted_samples))
    # print('length of data',len(dataset))

    weights = torch.FloatTensor(weights)
    

    return weights
weights_list=[]
for i in range(1,7):
    weights_list.append(BIO_train[i])
# weights1 = weight_classes(BIO_train1)
# weights2 = weight_classes(BIO_train2)
# weights3 = weight_classes(BIO_train3)
# weights4 = weight_classes(BIO_train4)
# weights5 = weight_classes(BIO_train5)
# weights6 = weight_classes(BIO_train6)




# weight_samples1 = weight_classes(BIO_train1)
# weight_samples2 = weight_classes(BIO_train2)
# weight_samples3 = weight_classes(BIO_train3)
# weight_samples4 = weight_classes(BIO_train4)
# weight_samples5 = weight_classes(BIO_train5)
# weight_samples6 = weight_classes(BIO_train6)

# weight_samples1= torch.DoubleTensor(weight_samples1)   
# sampler1 = torch.utils.data.sampler.WeightedRandomSampler(weight_samples1, len(weight_samples1)) 
# weight_samples2= torch.DoubleTensor(weight_samples2)   
# sampler2 = torch.utils.data.sampler.WeightedRandomSampler(weight_samples2, len(weight_samples2)) 
# weight_samples1= torch.DoubleTensor(weight_samples3)   
# sampler3 = torch.utils.data.sampler.WeightedRandomSampler(weight_samples3, len(weight_samples3)) 
# weight_samples1= torch.DoubleTensor(weight_samples4)   
# sampler4 = torch.utils.data.sampler.WeightedRandomSampler(weight_samples4, len(weight_samples4)) 
# weight_samples1= torch.DoubleTensor(weight_samples5)   
# sampler5 = torch.utils.data.sampler.WeightedRandomSampler(weight_samples5, len(weight_samples5)) 
# weight_samples1= torch.DoubleTensor(weight_samples6)   
# sampler6 = torch.utils.data.sampler.WeightedRandomSampler(weight_samples6, len(weight_samples6)) 


# # trainloader = DataLoader(BIO_train, shuffle=False, sampler=sampler,batch_size=BATCH_SIZE)


# trainloader1 = DataLoader(BIO_train1, shuffle=False,batch_size=BATCH_SIZE,sampler=sampler1)
# trainloader2 = DataLoader(BIO_train2, shuffle=False,batch_size=BATCH_SIZE,sampler=sampler2)
# trainloader3 = DataLoader(BIO_train3, shuffle=False,batch_size=BATCH_SIZE,sampler=sampler3)
# trainloader4 = DataLoader(BIO_train4, shuffle=False,batch_size=BATCH_SIZE,sampler=sampler4)
# trainloader5 = DataLoader(BIO_train5, shuffle=False,batch_size=BATCH_SIZE,sampler=sampler5)
# trainloader6 = DataLoader(BIO_train6, shuffle=False,batch_size=BATCH_SIZE,sampler=sampler6)

trainloader1 = DataLoader(BIO_train1, shuffle=False,batch_size=BATCH_SIZE)
trainloader2 = DataLoader(BIO_train2, shuffle=False,batch_size=BATCH_SIZE)
trainloader3 = DataLoader(BIO_train3, shuffle=False,batch_size=BATCH_SIZE)
trainloader4 = DataLoader(BIO_train4, shuffle=False,batch_size=BATCH_SIZE)
trainloader5 = DataLoader(BIO_train5, shuffle=False,batch_size=BATCH_SIZE)
trainloader6 = DataLoader(BIO_train6, shuffle=False,batch_size=BATCH_SIZE)

valloader1 = DataLoader(BIO_val1, shuffle=False,batch_size=BATCH_SIZE)
valloader2 = DataLoader(BIO_val2, shuffle=False,batch_size=BATCH_SIZE)
valloader3 = DataLoader(BIO_val3, shuffle=False,batch_size=BATCH_SIZE)
valloader4 = DataLoader(BIO_val4, shuffle=False,batch_size=BATCH_SIZE)
valloader5 = DataLoader(BIO_val5, shuffle=False,batch_size=BATCH_SIZE)
valloader6 = DataLoader(BIO_val6, shuffle=False,batch_size=BATCH_SIZE)

testloader1 = DataLoader(BIO_test1, shuffle=False,batch_size=BATCH_SIZE)
testloader2 = DataLoader(BIO_test2, shuffle=False,batch_size=BATCH_SIZE)
testloader3 = DataLoader(BIO_test3, shuffle=False,batch_size=BATCH_SIZE)
testloader4 = DataLoader(BIO_test4, shuffle=False,batch_size=BATCH_SIZE)
testloader5 = DataLoader(BIO_test5, shuffle=False,batch_size=BATCH_SIZE)
testloader6 = DataLoader(BIO_test6, shuffle=False,batch_size=BATCH_SIZE)


## with no sampler
# trainloader = DataLoader(BIO_train, batch_size=32, shuffle=True)
# valloader = DataLoader(BIO_val, batch_size=32,shuffle=True)
# testloader = DataLoader(BIO_test, batch_size=32,shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())
# model = Network().to(device)


################# MODEL1#####################
# model = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, numb_class1)

model = Network(output_dim=numb_class1)

model = model.to(device)
model.eval()

weights1 = weights1.to(device)
criterion1 = nn.CrossEntropyLoss(weight=weights_list[0]) # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer1 = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


################# MODEL2#####################
# model2 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
# num_ftrs = model2.fc.in_features
# model2.fc = nn.Linear(num_ftrs, numb_class2)

model2 = Network(output_dim=numb_class2)

model2 = model2.to(device)
model2.eval()
weights2 = weights2.to(device)
criterion2 = nn.CrossEntropyLoss(weight=weights2) # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


################# MODEL3#####################
# model3 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
# num_ftrs = model3.fc.in_features
# model3.fc = nn.Linear(num_ftrs, numb_class3)


model3 = Network(output_dim=numb_class3)

model3 = model3.to(device)
model3.eval()
weights3 = weights3.to(device)
criterion3 = nn.CrossEntropyLoss(weight=weights3) # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer3 = optim.Adam(model3.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


################# MODEL4#####################
# model4 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
# num_ftrs = model4.fc.in_features
# model4.fc = nn.Linear(num_ftrs, numb_class4)

model4 = Network(output_dim=numb_class4)

model4 = model4.to(device)
model4.eval()
weights4 = weights4.to(device)
criterion4 = nn.CrossEntropyLoss(weight=weights4) # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer4 = optim.Adam(model4.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength

################# MODEL5#####################
# model5 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
# num_ftrs = model5.fc.in_features
# model5.fc = nn.Linear(num_ftrs, numb_class5)

model5 = Network(output_dim=numb_class5)

model5 = model5.to(device)
model5.eval()
weights5 = weights5.to(device)
criterion5 = nn.CrossEntropyLoss(weight=weights5) # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer5 = optim.Adam(model5.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


################# MODEL6#####################
# model6 = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
# num_ftrs = model6.fc.in_features
# model6.fc = nn.Linear(num_ftrs, numb_class6)


model6 = Network(output_dim=numb_class6)

model6 = model5.to(device)
model6.eval()
weights6 = weights6.to(device)
criterion6 = nn.CrossEntropyLoss(weight=weights6) # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer6 = optim.Adam(model6.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength


            ####
            # 1 : LW, 2:RA, 3:RD, 4:SA, 5:SD, 6:Stand
            # MODEL1: LW(1)-> LW(1), RA(2), RD(3), SA(4), SD(5), Stand(6) 
            # MODEL2: RA(2)-> LW(1), RA(2)
            # MODEL3: RD(3)-> LW(1), RD(3)
            # MODEL4: SA(4)-> LW(1), SA(4)
            # MODEL5: SD(5)-> LW(1), SD(5)
            # MODEL6: Stand(6)-> LW(1), SD(6)

def train(model, criterion, optimizer, loader, valloader, num_epoch = 20,label_no=None): # Train the model
    loss_history=[]
    val_history=[]
    print("Start training...")
    model.train() # Set the model to training mode
    # model2.train()

    for i in range(num_epoch):
        model.train()
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            label = label -1 # indexing start from 1 (removing sitting conditon)
            if label_no>2:
                label = label/torch.LongTensor([label_no-1])
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weightspre
            loss_history.append(np.mean(running_loss))

            # elif label ==2:
        _,_,val_acc = evaluate(model, valloader,label_no)
        val_history.append(val_acc)
        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc)) # Print the average loss for this epoch
    print("Done!")
    return loss_history, val_history

def evaluate(model, loader,label_no): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            label = label-1
            if label_no>2:
                label = label/torch.LongTensor([label_no-1])
            # if label_no ==6:
            #     pdb.set_trace()
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    if len(loader.dataset) != 0:
        acc = correct/len(loader.dataset)
        print("Evaluation accuracy: {}".format(acc))
    else:
        acc =0
        print('empty dataset!!', label_no)
    return correct, len(loader.dataset), acc

loss_history, val_history =train(model, criterion1, optimizer1, trainloader1,valloader1, num_epoch,label_no=1)
loss_history2, val_history2 =train(model2, criterion2, optimizer2, trainloader2,valloader2, num_epoch,label_no=2)
loss_history3, val_history3 =train(model3, criterion3, optimizer3, trainloader3, valloader3,num_epoch,label_no=3)
loss_history4, val_history4 =train(model4, criterion4, optimizer4, trainloader4, valloader4,num_epoch,label_no=4)
loss_history5, val_history5 =train(model5, criterion5, optimizer5, trainloader5, valloader5,num_epoch,label_no=5)
loss_history6, val_history6 =train(model6, criterion6, optimizer6, trainloader6, valloader6,num_epoch,label_no=6)


print("Evaluate on test set")
corr1, len_data1,_ =evaluate(model, testloader1,label_no=1)
corr2, len_data2,_ =evaluate(model2, testloader2,label_no=2)
corr3, len_data3,_ =evaluate(model3, testloader3,label_no=3)
corr4, len_data4,_ =evaluate(model4, testloader4,label_no=4)
corr5, len_data5,_ =evaluate(model5, testloader5,label_no=5)
corr6, len_data6,_ =evaluate(model6, testloader6,label_no=6)

corr_total = corr1+corr2+corr3+corr4+corr5+corr6
len_data_total= len_data1+len_data2+len_data3+len_data4+len_data5+len_data6
acc_total = corr_total/len_data_total
print("Total Evaluation accuracy: {}".format(acc_total))

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



