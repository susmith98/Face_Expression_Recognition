#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import cv2
import numpy as np
import os


base_dir=r'D:\ProjectML'
f_p=base_dir+r'\fer.csv'

ds=pd.read_csv(f_p)

emotion={0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise"}

# CREATING FOLDERS FOR EACH OF THE EXPRESSION INSIDE THE TRAIN FOLDER
for i in range(0,6):
    os.mkdir(os.path.join(base_dir,r'train',emotion[i]))

# LOADING TRAINING PICTURES FROM DATA FRAME TO RESPECTIVE FOLDERS
c=0
for i in range(6):
    temp=ds[(ds['emotion']==i) & (ds['Usage']=='Training')]
    for k in range(len(temp)):
        pix=temp.iloc[k]['pixels']
        img=np.array([int(j) for j in pix.split(' ')])
        img=np.reshape(img,(48,48))
        cv2.imwrite(os.path.join(base_dir,r'train',emotion[temp.iloc[k]['emotion']],(str)(c)+'.jpg'),img)
        c=c+1
    

# CREATING FOLDERS FOR EACH OF THE EXPRESSION INSIDE THE VALID FOLDER(FOLDER FOR VALIDATION DATA)
for i in range(0,6):
    os.mkdir(os.path.join(base_dir,r'valid',emotion[i]))

c=0
for i in range(6):
    temp=ds[(ds['emotion']==i) & (ds['Usage']=='PublicTest')]
    for k in range(len(temp)):
        pix=temp.iloc[k]['pixels']
        img=np.array([int(j) for j in pix.split(' ')])
        img=np.reshape(img,(48,48))
        cv2.imwrite(os.path.join(base_dir,r'valid',emotion[temp.iloc[k]['emotion']],(str)(c)+'.jpg'),img)
        c=c+1
    

