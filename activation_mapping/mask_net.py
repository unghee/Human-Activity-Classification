import torch
from torch import nn

#dummy networks to help visualize activations of the first conv layer

class Network(nn.Module):
    def __init__(self,INPUT_NUM,NUMB_CLASS):
        super().__init__()
        self.sclayer1 = nn.Sequential(
            nn.Conv2d(INPUT_NUM, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.sclayer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear( 4096, 2000)
        self.fc1 = nn.Linear( 6144, 2000)
        self.fc2 = nn.Linear(2000, NUMB_CLASS)


    def forward(self,x):
        #only look at first layer, other layers are too coarse
        x = self.sclayer1(x)

        return x

class FigureNetwork(nn.Module):
    def __init__(self,INPUT_NUM,NUMB_CLASS):
        super().__init__()
        self.sclayer1 = nn.Sequential(
            nn.Conv2d(INPUT_NUM, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.sclayer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear( 4096, 2000)
        self.fc1 = nn.Linear( 6144, 2000)
        self.fc2 = nn.Linear(2000, NUMB_CLASS)


    def forward(self,x):
        x = self.sclayer1(x)
        xtemp = torch.clone(x)
        x = self.sclayer2(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)



        return x, xtemp

        return x

