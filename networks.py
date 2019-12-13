import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split



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