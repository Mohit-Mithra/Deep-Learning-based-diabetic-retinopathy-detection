import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.applications import inception_resnet_v2
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

import scipy
import tensorflow as tf
#from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import layers
import keras

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import tensorflow as tf
tf.device('/cpu:0')
#from network import *
from load_data import *
from preprocess import *
from model import *
from Inception_Resnet_model import *
from pre_process_test_images import *
from simple_model import *

np.random.seed(2019)
tf.set_random_seed(2019) 

print("Called Loaddata")
train_df, test_df = load_data()

print("Called Preprocessing")
x_train, x_val, y_train, y_val = preprocess_data(train_df, test_df)


#model = build_model_Inception_Resnet()                   #For Densenet
#model = build_model_Inception_Resnet()                 #For Inception-Resnet
model = build_simple_model()                            #For simple model

model.summary()


BATCH_SIZE = 32

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.50,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )


# Using original generator
data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)

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

kappa_metrics = Metrics()

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[kappa_metrics]
)

with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()

plt.plot(kappa_metrics.val_kappas)

y_pred = model.predict(x_val) 
print("0")
print(y_pred)
#print(accuracy_score(y_val, y_pred))
# y_test = model.predict(x_test) > 0.7
# y_test = y_test.astype(int).sum(axis=1) - 1

# test_df['diagnosis'] = y_test
# test_df.to_csv('submission.csv',index=False)

print(y_val)
print("1")

y_val_1d = np.ndarray(y_val.shape[0])
for i in range(y_val.shape[0]):
    y_val_1d[i] = y_val[i].sum()


y_pred_1d = np.ndarray(y_pred.shape[0])
for i in range(y_pred.shape[0]):
    y_pred_1d[i] = y_pred[i].sum()

print(confusion_matrix(y_val_1d, y_pred_1d))


test_df = pd.read_csv('IDRiD_Disease Grading_Testing Labels.csv')
test_df.columns = ["id_code", "diagnosis", "discard"]


x_test, y_test = preprocess_test_data(test_df)
y_test_pred = model.predict(x_test) > 0.5
print(accuracy_score(y_test, y_test_pred))

print()
y_test_1d = np.ndarray(y_test.shape[0])
for i in range(y_test.shape[0]):
    y_test_1d[i] = y_test[i].sum()


y_test_pred_1d = np.ndarray(y_test_pred.shape[0])
for i in range(y_test_pred.shape[0]):
    y_test_pred_1d[i] = y_test_pred[i].sum()

print(confusion_matrix(y_test_1d, y_test_pred_1d))