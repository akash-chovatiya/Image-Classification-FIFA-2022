# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 10:12:11 2023

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

#print('Total number of images in the dataset:', len(players_df))
#print('Total number of players in the dataset: ', players_df['label'].value_counts())

import PIL
import numpy as np
images = []
labels = []
for i in classes:
    player_images = sorted(os.listdir(os.path.join(parentdir,"dataset", "Images", "Group C", "Argentina Players",i)))
    
    for j in player_images:
        image_path = os.path.join(parentdir,"dataset", "Images", "Group C", "Argentina Players",i,j)
        image = PIL.Image.open(image_path)
        image = image.resize((300,300))
        image = np.asarray(image)
        images.append(image)
        labels.append(i)

image_data = np.array(images)
image_data = image_data.astype('float32')/255.0
print(image_data.shape)

import sklearn.preprocessing as sns
label_encode = sns.LabelEncoder()
encoded_label = label_encode.fit_transform(players_df['label'].values)
# label_encode.fit(players_df['label'].values)
# classes = list(label_encode.classes_)
encoded_label = encoded_label.reshape(-1, 1)
onehotencoder = sns.OneHotEncoder()
y = onehotencoder.fit_transform(encoded_label)
print(y.shape)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
image_data, y = shuffle(image_data, y, random_state = 11)
train_x, test_x, train_y, test_y = train_test_split(image_data, y, test_size = 0.1, random_state = 111)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)