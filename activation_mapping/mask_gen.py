import torch
import numpy as np
from mask_net import Network


#creates a mask that can be used to visualize some activations of the network

#parameters
NUMB_CLASS = 5
device = "cpu"

#make a dummy model
model = Network(52,NUMB_CLASS)
model = model.to(device)

#initialize a mask
mask = np.zeros((10,50,5,25))

for i in range(10):
    for j in range(50):
        x = np.zeros((1, 52,10,50)).astype(float)
        #set an initial value to nan
        x[:,:,i,j] = np.nan
        #run it through the model
        x = torch.from_numpy(x)
        x = x.to(device, dtype=torch.float)
        xtemp = model.forward(x)
        #save all output nans to the mask
        for k in range(5):
            for l in range (25):
                if not np.isfinite(xtemp.detach()[0][0][k][l]):
                    mask[i][j][k][l] = 1

print("Saving mask...")
np.save("mask", mask)



