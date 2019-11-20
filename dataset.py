import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd

class EnableDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):