# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:41:28 2023

@author: Chovatiya
"""

import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir)

grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0,grandgrandparentdir)

import pandas as pd

classes = sorted(os.listdir(os.path.join(parentdir,"dataset", "Images","Group C","Argentina Players")))
players_data = []
for i in classes:
    player_images = sorted(os.listdir(os.path.join(parentdir,"dataset", "Images", "Group C", "Argentina Players",i)))
    
    for j in player_images:
        image_path = os.path.join(parentdir,"dataset", "Images", "Group C", "Argentina Players",i,j)
        players_data.append((i,image_path))

players_df = pd.DataFrame(data=players_data, columns=['label','image'])

import sklearn.preprocessing as sns
from sklearn.model_selection import StratifiedShuffleSplit
import torch,torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from LoadDataset import FIFA2022
from model import FIFAmodel
#from model_vgg16 import FIFAmodel
from validator import evaluate
import torch.nn as nn

label_encode = sns.LabelEncoder()
encoded_label = label_encode.fit_transform(players_df['label'].values)
encoded_label = encoded_label.reshape(-1, 1)
# onehotencoder = sns.OneHotEncoder()
# y = onehotencoder.fit_transform(encoded_label)
# y = torch.as_tensor(torch.from_numpy(y.toarray()),dtype=torch.long)
y = torch.as_tensor(torch.from_numpy(encoded_label), dtype=torch.long)

def dataset_split(x, y, size):
    stratSplit = StratifiedShuffleSplit(n_splits=1,test_size=size, random_state=50)
    stratSplit.get_n_splits(x,y)
    for i, (train_index, test_index) in enumerate(stratSplit.split(x,y)):
        x_train = players_df.iloc[train_index]
        x_test = players_df.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = dataset_split(players_df['image'].values, y, 0.2)

x_train, y_train, x_val, y_val = dataset_split(x_train['image'].values, y_train, 0.2)

train_dataset = FIFA2022(x_train, y_train)
val_dataset = FIFA2022(x_val, y_val)
test_dataset = FIFA2022(x_test, y_test)

batch_size = 6

train_dl = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size, shuffle=True, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, shuffle=True, pin_memory=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
#DEVICE = torch.device('cpu')
model = FIFAmodel(4)
# model = torchvision.models.vgg16(pretrained=False)
# num_features = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_features, 4)
model.to(DEVICE)
#model.load_state_dict(torch.load(os.path.join(parentdir,"dataset","results","FIFA2022.pth")))
params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.SGD(params, lr = 0.0001, weight_decay = 0.005, momentum = 0.9)
optimizer = torch.optim.Adam(params, lr = 0.0001, weight_decay = 0.005)
criterion = nn.CrossEntropyLoss()
  
num_epochs = 100
history = []

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for images, targets in train_dl:
        optimizer.zero_grad()
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        out = model(images)
        #loss = F.cross_entropy_loss(out, targets.view(-1))
        loss = criterion(out, targets.view(-1))
        train_losses.append(loss)
        del images, targets
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    result = evaluate(model, val_dl)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    history.append(result)
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))
    
    torch.save(model.state_dict(), os.path.join(parentdir,"dataset","results","FIFA2022.pth"))