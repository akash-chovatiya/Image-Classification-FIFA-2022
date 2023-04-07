# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:00:30 2023

@author: Chovatiya
"""

import PIL
from torch.utils.data import Dataset
import torchvision

class FIFA2022(Dataset):
    def __init__(self, dataframe, labels):
        super().__init__()
        self.dataframe = dataframe
        self.labels = labels
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            #torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            #torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            torchvision.transforms.ToTensor()])
    
    def __getitem__(self, idx):
        path = self.dataframe.iloc[idx][1]
        image = PIL.Image.open(path)
        label = self.labels[idx]
        
        return (self.transforms(image))/255.0, label
    
    def __len__(self):
        return len(self.dataframe)