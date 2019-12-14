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
numb_class= [6,2,2,2,2,2]


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

BIO_trains=[]
BIO_vals=[]
BIO_tests=[]
for i in range(6):
    BIO_trains.append(EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1,48),window_size=500,processed=True,label=i+1))
    BIO_vals.append(EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(48,49),window_size=500,processed=True,label=i+1))
    BIO_tests.append(EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(49,50),window_size=500,processed=True,label=i+1))
    


# ## saving dataset a file

save_object(BIO_trains, 'BIO_trains.pkl')
save_object(BIO_vals, 'BIO_vals.pkl')
save_object(BIO_tests, 'BIO_tests.pkl')

# ## load from saved files
# with open('BIO_trains.pkl', 'rb') as input:
#     BIO_trains = pickle.load(input)
# with open('BIO_vals.pkl', 'rb') as input:
#     BIO_vals = pickle.load(input)
# with open('BIO_tests.pkl', 'rb') as input:
#     BIO_tests = pickle.load(input)


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
            # weights.append(sum_classes/idx)
            weights.append(1)
        else:
            continue
    print(weights)


    weights = torch.FloatTensor(weights)
    

    return weights

weights_list=[]
for i in range(6):
    weights_list.append(weight_classes(BIO_trains[i]))

trainloaders=[]
valloaders=[]
testloaders=[]
for i in range(6):
    trainloaders.append(DataLoader(BIO_trains[i], shuffle=False,batch_size=BATCH_SIZE))
    valloaders.append(DataLoader(BIO_vals[i], shuffle=False,batch_size=BATCH_SIZE))
    testloaders.append(DataLoader(BIO_tests[i], shuffle=False,batch_size=BATCH_SIZE))


device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())

################# MODELS#####################

models =[]
optimizers =[]
criterions =[]

for i in range(6):
    # # model = torch.hub.load('pytorch/vision:v0.4.2', 'resnext50_32x4d', pretrained=True) # use resnet
    # # num_ftrs = model.fc.in_features
    # # model.fc = nn.Linear(num_ftrs, numb_class1)
    model = Network(output_dim=numb_class[i])
    model = model.to(device)
    model.eval()
    models.append(model)
    weights_list[i] = weights_list[i].to(device)
    criterions.append(nn.CrossEntropyLoss(weight=weights_list[i])) # Specify the loss layer
    # TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
    optimizers.append(optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
    del model


#### MODE-SPECIFIC CLASSIFIER SCHEME
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

loss_historys=[]
val_historys=[]
for i in range(6):
    loss_history, val_history =train(models[i], criterions[i], optimizers[i], trainloaders[i], valloaders[i], num_epoch, label_no=i+1)
    loss_historys.append(loss_history)
    val_historys.append(val_history)


print("Evaluate on test set")


corrs=[]
len_datas=[]
for i in range(6):
    corr, len_data,_ =evaluate(models[i], testloaders[i],label_no=i+1)
    corrs.append(corr)
    len_datas.append(len_data)


corr_total =0
len_data_total=0
for i in range(6):
    corr_total = corrs[i] + corr_total
    len_data_total = len_data_total + len_datas[i]


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



