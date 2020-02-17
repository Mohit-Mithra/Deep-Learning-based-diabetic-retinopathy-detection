# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
#from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import layers
import keras

#%matplotlib inline


np.random.seed(2019)
tf.set_random_seed(2019)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()

train_df['diagnosis'].hist()
train_df['diagnosis'].value_counts()

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im


N = train_df.shape[0]
x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

j = 0
for i, image_id in enumerate((train_df['id_code'])):
     
    if(j%500 == 0):
        print(j)
    x_train[i, :, :, :] = preprocess_image('train_images/'+image_id+'.png')
    j = j+1;

# N = test_df.shape[0]
# x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

# j = 0
# for i, image_id in enumerate((test_df['id_code'])):
#     if(j%500 == 0):
#         print(j)
#     x_test[i, :, :, :] = preprocess_image('test_images/'+image_id+'.png')
#     j = j+1;


y_train = pd.get_dummies(train_df['diagnosis']).values

print(x_train.shape)
print(y_train.shape)
#print(x_test.shape)

y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_multi, 
    test_size=0.15, 
    random_state=2019
)

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print("val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return
    
densenet = DenseNet121(
    weights='DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dropout(0.5))
    
    # model.add(layers.Conv2D(128,(3,3),activation='relu'))
    # model.add(layers.MaxPooling2D((2,2)))
    # model.add(layers.Dropout(0.5))
    
    # model.add(layers.Conv2D(64,(3,3),activation='relu'))
    # model.add(layers.MaxPooling2D((2,2)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model

model = build_model()
model.summary()

BATCH_SIZE = 32

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )


# Using original generator
data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)


kappa_metrics = Metrics()

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=12,
    validation_data=(x_val, y_val),
    callbacks=[kappa_metrics]
)

with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()

plt.plot(kappa_metrics.val_kappas)


# y_test = model.predict(x_test) > 0.7
# y_test = y_test.astype(int).sum(axis=1) - 1

# test_df['diagnosis'] = y_test
# test_df.to_csv('submission.csv',index=False)

test_df.head



