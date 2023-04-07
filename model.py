# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:56:07 2023

@author: Chovatiya
"""

import torch.nn as nn

class FIFAmodel(nn.Module):
    def __init__(self, num_classes):
        super(FIFAmodel, self).__init__()
        in_channels = 3
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 64 x 64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 32 x 32

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 16 x 16
            
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256*16*16,1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))
        
    def forward(self, xb):
        return self.network(xb)