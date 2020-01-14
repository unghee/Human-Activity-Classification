import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms , utils
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import EnableDataset

from tempfile import TemporaryFile
import pickle
from skimage import io, transform
import pdb

from networks import Network

########## SETTINGS  ########################

## for N way classifieres
numb_class= [5,2,2,2,2]
len_class = len(numb_class)

# numb_class = 6
SAVE_MODEL = False
num_epoch = 60

BATCH_SIZE = 32
LEARNING_RATE = 1e-4*0.8
WEIGHT_DECAY = 1e-4


############################################

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# Load the dataset and train, val, test splits
print("Loading datasets...")

BIO_trains=[]
BIO_tests=[]
events = ['RC','RT','LT','LC']
# for i in range(len_class):
#     for event in events:
#         BIO_trains.append(EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(1,46),window_size=500,label=i+1,event_label=event))
#         BIO_tests.append(EnableDataset(subject_list= ['156','185','186','188','189','190', '191', '192', '193', '194'],data_range=(46,50),window_size=500,label=i+1,event_label=event))
#         print("label_{}_event_{} loaded".format(i,event))

# for i in range(len_class):
#     for event in events:
#         BIO_trains.append(EnableDataset(subject_list= ['156'],data_range=(1,4),window_size=500,label=i+1,event_label=event))
#         BIO_tests.append(EnableDataset(subject_list= ['156'],data_range=(46,50),window_size=500,label=i+1,event_label=event))
#         print("label_{}_event_{} loaded".format(i,event))

# save_object(BIO_trains,'BIO_trains_reduced_size.pkl')
# save_object(BIO_tests,'BIO_tests_reduced_size.pkl')

with open('BIO_trains_reduced_size.pkl', 'rb') as input:
    BIO_trains = pickle.load(input)
with open('BIO_tests_reduced_size.pkl', 'rb') as input:
    BIO_tests = pickle.load(input)



## check the class distribution

def weight_classes(dataset):
    trainloader = DataLoader(dataset, shuffle=False,batch_size=BATCH_SIZE)
    classes = [0,0,0,0,0,0,0]
    for data, labels in trainloader:
        for x in range(labels.size()[0]):
            classes[labels[x]] +=1
    # print(classes)

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

weights_list=[]
j = 0
for i in range(len_class):
    for event in events:
        weights_list.append(weight_classes(BIO_trains[j]))
        j+=1

validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
train_samplers=[]
valid_samplers=[]
val_lens=[]
j = 0
for i in range(len_class):
    for event in events:
        dataset_size = len(BIO_trains[j])
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        val_len=len(val_indices)
        val_lens.append(val_len)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_samplers.append(train_sampler)
        valid_samplers.append(valid_sampler)
        j +=1




trainloaders=[]
valloaders=[]
testloaders=[]
j = 0
for i in range(len_class):
    for event in events:
        trainloaders.append(DataLoader(BIO_trains[j], shuffle=False,batch_size=BATCH_SIZE,sampler = train_samplers[j]))
        valloaders.append(DataLoader(BIO_trains[j], shuffle=False,batch_size=BATCH_SIZE,sampler = valid_samplers[j]))
        testloaders.append(DataLoader(BIO_tests[j], shuffle=False,batch_size=BATCH_SIZE))
        j +=1


device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())

################# MODELS#####################

models =[]
optimizers =[]
criterions =[]
j = 0
for i in range(len_class):
    for event in events:
            model = Network(output_dim=numb_class[i])
            model = model.to(device)
            model.eval()
            models.append(model)
            weights_list[j] = weights_list[j].to(device)
            criterions.append(nn.CrossEntropyLoss(weight=weights_list[j]))
            optimizers.append(optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY))
            del model
            j +=1


#### MODE-SPECIFIC CLASSIFIER SCHEME
# 1 : LW, 2:RA, 3:RD, 4:SA, 5:SD, 6:Stand
# MODEL1: LW(1)-> LW(1), RA(2), RD(3), SA(4), SD(5), Stand(6)
# MODEL2: RA(2)-> LW(1), RA(2)
# MODEL3: RD(3)-> LW(1), RD(3)
# MODEL4: SA(4)-> LW(1), SA(4)
# MODEL5: SD(5)-> LW(1), SD(5)
# MODEL6: Stand(6)-> LW(1), SD(6) ## PAPER DOES NOT DO THIS

def train(model, criterion, optimizer, loader, valloader, num_epoch = 20,label_no=None, event=None,vallen=None): # Train the model
    loss_history=[]
    val_history=[]
    print("Start training...")
    model.train()

    for i in range(num_epoch):
        model.train()
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label -1 # indexing start from 1 (removing sitting conditon)
            if label_no>2:
                label = label/torch.LongTensor([label_no-1])
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            loss_history.append(np.mean(running_loss))
        _,_,val_acc = evaluate(model, valloader,label_no,vallen)
        val_history.append(val_acc)
        print("Epoch {} loss:{} val_acc:{}".format(i+1,np.mean(running_loss),val_acc)) # Print the average loss for this epoch


    if SAVE_MODEL:
        torch.save(model, "model_class{}_{}.pth".format(label_no,event))
        print('model saved label:{} event:{}'.format(label_no,event))
    print("Done!")
    return loss_history, val_history

def evaluate(model, loader,label_no,vallen=None): # Evaluate accuracy on validation / test set
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch, label in tqdm(loader):
            batch = batch.to(device)

            pred = model(batch)
            label = label-1
            if label_no>2:
                label = label/torch.LongTensor([label_no-1])
            label = label.to(device)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    if len(loader.dataset) != 0 and vallen != None:
        acc = correct/vallen if vallen != 0 else 0
        print("Evaluation accuracy: {}".format(acc))
    elif vallen == None:
        acc = correct/len(loader.dataset) if len(loader.dataset) != 0 else 0
        print("Evaluation accuracy: {}".format(acc))
    else:
        acc =0
        print('empty dataset!!', label_no)
    return correct, len(loader.dataset), acc

loss_historys=[]
val_historys=[]
# j = 15
j = 0
for i in range(len_class):
    for event in events:

        # print(event)
        loss_history, val_history =train(models[j], criterions[j], optimizers[j], trainloaders[j], valloaders[j], num_epoch, label_no=i+1,event=event,vallen=val_lens[j])
        loss_historys.append(loss_history)
        val_historys.append(val_history)
        print('*****class{}_{}*********'.format(i+1,event))
        j +=1


print("Evaluate on test set")


corrs=[]
len_datas=[]
j =0
for i in range(len_class):
    for event in events:
        corr, len_data,_ =evaluate(models[j], testloaders[j],label_no=i+1 )
        corrs.append(corr)
        len_datas.append(len_data)
        print('Evlauation on class{}_{}*********'.format(i+1,event))
        j +=1


corr_total =0
len_data_total=0
j =0 
for i in range(len_class):
    for event in events:
        corr_total = corrs[j] + corr_total
        len_data_total = len_data_total + len_datas[j]
        j +=1


acc_total = corr_total/len_data_total
print("Total Evaluation accuracy: {}".format(acc_total))

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


