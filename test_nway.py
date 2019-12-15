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
numb_class= [6,2,2,2,2]
# numb_class= [6]
len_class = len(numb_class)

# numb_class = 6
SAVE_MODEL = True
num_epoch = 4

BATCH_SIZE = 32
LEARNING_RATE = 1e-4*0.8
WEIGHT_DECAY = 1e-4


############################################



for i in range(len_class):
    model=torch.load('model_class{}.pth'.format(i))
    models.append(model)
    del model

with open('BIO_tests.pkl', 'rb') as input:
    BIO_tests = pickle.load(input)

for i in range(len_class):
    testloaders.append(DataLoader(BIO_tests[i], shuffle=False,batch_size=BATCH_SIZE))

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print('GPU USED?',torch.cuda.is_available())

################# MODELS#####################



#### MODE-SPECIFIC CLASSIFIER SCHEME
# 1 : LW, 2:RA, 3:RD, 4:SA, 5:SD, 6:Stand
# MODEL1: LW(1)-> LW(1), RA(2), RD(3), SA(4), SD(5), Stand(6) 
# MODEL2: RA(2)-> LW(1), RA(2)
# MODEL3: RD(3)-> LW(1), RD(3)
# MODEL4: SA(4)-> LW(1), SA(4)
# MODEL5: SD(5)-> LW(1), SD(5)
# MODEL6: Stand(6)-> LW(1), SD(6) ## PAPER DOES NOT DO THIS

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


print("Evaluate on test set")


corrs=[]
len_datas=[]
for i in range(len_class):
    # for event in events:
        corr, len_data,_ =evaluate(models[i], testloaders[i],label_no=i+1)
        corrs.append(corr)
        len_datas.append(len_data)


corr_total =0
len_data_total=0
# for i in range(len_class*2):
for i in range(len_class):
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



