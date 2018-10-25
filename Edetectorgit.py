#!/usr/bin/env python
# coding: utf-8


#IMPORTING REQUIRED LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
from keras import models,layers,regularizers,optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import load_model,model_from_json
import pandas as pd
from sklearn.metrics import confusion_matrix



# LOADING THE PRETRAINED VGG MODEL WITH IMAGENET WEIGHTS
# INCLUDE_TOP = FALSE (ONLY TAKING CONVOLUTIONAL BASE)
Model=applications.VGG16(include_top=False,weights='imagenet',input_shape=(48,48,3))
Model.summary()


#CREATING A SEQUENTIAL MODEL OUT OF VGG FOR FEATURE EXTRACTION
model=models.Sequential()

# ADDING ONLY 11 LAYERS OF VGG
for i in Model.layers[:11]:
    model.add(i)

# SETTING ALL THESE LAYERS AS NOT TRAINABLE
for i in model.layers:
    i.trainable=False


# CREATING A MODEL ON TOP OF VGG TO PROCESS THE FEATURES EXTRACTED OUT OF THE VGG MODEL
VGGtop=models.Sequential()
VGGtop.add(layers.Conv2D(512,(3,3),input_shape=(6,6,256),kernel_regularizer=regularizers.l2(0.01)))
VGGtop.add(layers.Conv2D(512,(3,3),kernel_regularizer=regularizers.l2(0.01)))
VGGtop.add(layers.MaxPool2D((2,2)))
VGGtop.add(layers.Flatten())
VGGtop.add(layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
VGGtop.add(layers.Dropout(0.5))
VGGtop.add(layers.Dense(32,activation='relu',kernel_regularizer=regularizers.l2(0.05)))
VGGtop.add(layers.Dense(6,activation='softmax'))

VGGtop.summary()

VGGtop.compile(optimizer=optimizers.SGD(lr=0.009),loss='categorical_crossentropy',metrics=['accuracy'])


train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.2,shear_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


train_datagenerator=train_datagen.flow_from_directory(os.path.join(r'D:\ProjectML',r'train'),target_size=(48,48),class_mode='categorical',batch_size=128)
valid_datagenetor=test_datagen.flow_from_directory(os.path.join(r'D:\ProjectML',r'valid'),target_size=(48,48),class_mode='categorical',batch_size=128)


# EXTRACTING FEATURES OF TRAINING DATA BY PASSING THEM THROUGH VGG MODEL
X_features=np.zeros(shape=(27520,6,6,256))
X_labels=np.zeros(shape=(27520,6))
i=0
for inputs_batch,labels_batch in train_datagenerator:
    X_features[i*128:(i+1)*128]=model.predict(inputs_batch,verbose=1)
    X_labels[i*128:(i+1)*128]=labels_batch
    i+=1
    if i*128>=27520:
        break
       

# EXTRACTING FEATURES OF VALIDATION DATA BY PASSING THEM THROUGH VGG MODEL
V_features=np.zeros(shape=(3584,6,6,256))
V_labels=np.zeros(shape=(3584,6))
i=0
for inputs_batch,labels_batch in valid_datagenetor:
    V_features[i*128:(i+1)*128]=model.predict(inputs_batch,verbose=1)
    V_labels[i*128:(i+1)*128]=labels_batch
    i+=1
    if i*128>=3584:
        break
       

# LIMITING THE SIZE OF VALIDATION FEATURES EXTRACTED(TO ADJUST THE BATCH SIZE AND STEPS PER EPOCH)
V_features=V_features[:3456]
V_labels=V_labels[:3456]


# CODE TO SAVE AND LOAD THE FEATURES
np.save(os.path.join(r'D:\ProjectML',r'X_features'),X_features)
np.save(os.path.join(r'D:\ProjectML',r'X_labels'),X_labels)
np.save(os.path.join(r'D:\ProjectML',r'V_features'),V_features)
np.save(os.path.join(r'D:\ProjectML',r'V_labels'),V_labels)

X_features=np.load(os.path.join(r'D:\ProjectML',r'X_features.npy'))
X_labels=np.load(os.path.join(r'D:\ProjectML',r'X_labels.npy'))
V_features=np.load(os.path.join(r'D:\ProjectML',r'V_features.npy'))
V_labels=np.load(os.path.join(r'D:\ProjectML',r'V_labels.npy'))


checkpoint=ModelCheckpoint(os.path.join(r'D:\ProjectML',r'Checkpointx.hdf5'),monitor='val_loss',mode='min',save_best_only=True,verbose=1)


# TRAINING THE MODEL
history=VGGtop.fit(X_features,X_labels,epochs=50,batch_size=128,validation_data=(V_features,V_labels),callbacks=[checkpoint],verbose=1)

# SAVING MODEL
open(r'VGGtopModel2.json','w').write(VGGtop.to_json())

# LOADING MODEL WEIGHTS(JUST CODE,ALREADY SAME WEIGHTS ARE LOADED)
VGGtop.load_weights(os.path.join(r'D:\ProjectML',r'Checkpointx.hdf5'))


# NOW THE PART OF DETECTING THE EMOTIONS IN REAL TIME USING WEB VIDEO 
import cv2
import dlib


emotion={0:'Angry',1:'Fear',2:'Happy',3:'Neutral',4:'Sad',5:'Surprise'}

fd=dlib.get_frontal_face_detector()


# DETECTING MULTIPLE FACES IN THE FRAME AND DETECTING FACE EMOTION FOR EVERY FACE
cam=cv2.VideoCapture(0)
while True:
    ret,img=cam.read()
    temp1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=fd(img,1)
    x=[]
    y=[]
    w=[]
    h=[]
    flag=False
    for i,f in enumerate(faces):
        flag=True
        a=f.left()
        b=f.top()
        x.append(a)
        y.append(b)
        w.append(f.right()-a)
        h.append(f.bottom()-b)
        
    #temp=img[y:y+h,x:x+w]
    if flag==True:
        for i in range(len(x)):
            cv2.imwrite(os.path.join(r'D:\ProjectML',r'temp.jpg'),temp1[y[i]:y[i]+h[i],x[i]:x[i]+w[i]])
            temp=cv2.imread(os.path.join(r'D:\ProjectML',r'temp.jpg'))
            temp=cv2.resize(temp,(48,48))
            temp=temp/255
            p=VGGtop.predict(model.predict(temp.reshape(1,48,48,3)))
            p=p.argmax()
            cv2.putText(img,emotion[p],(x[i]-20,y[i]),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255))
    
        
    cv2.imshow('IMAGE',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
        
        
cv2.destroyAllWindows()    
cam.release()







