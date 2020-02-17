import os

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
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import StratifiedShuffleSplit

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


def preprocess_data(train_df, test_df):
    N = train_df.shape[0]
    x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

    j = 0
    #for i, image_id in enumerate((train_df['id_code'])):
         
        #if(j%500 == 0):
        #    print(j)

        #img = preprocess_image('train_images/'+image_id+'.png')
        ##img_array = img_to_array(img)
        ##save_img('processed_train_images/'+image_id+'.png', img_array)

        #img.save('processed_train_images/'+image_id+'.png')

        #j = j+1;
        ##print(j)
        ##if j == 100:
        ##	break

        ##x_train[i, :, :, :] = preprocess_image('train_images/'+image_id+'.png')
        ##j = j+1;

    # N = test_df.shape[0]
    # x_test = np.empty((N, 224, 224, 3), dtyp  e=np.uint8)

    # j = 0
    # for i, image_id in enumerate((test_df['id_code'])):
    #     if(j%500 == 0):
    #         print(j)
    #     x_test[i, :, :, :] = preprocess_image('test_images/'+image_id+'.png')
    #     j = j+1;

    for i, image_id in enumerate((train_df['id_code'])):
         

        x_train[i, :, :, :] = Image.open('processed_train_images/'+image_id+'.png')

    train_df['diagnosis'] = train_df['diagnosis'] > 0
    train_df['diagnosis'] = train_df['diagnosis'] * 1.0

    y_train = pd.get_dummies(train_df['diagnosis']).values

    print(x_train.shape)
    print(y_train.shape)
    #print(x_test.shape)
    y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
    y_train_multi[:, 1] = y_train[:, 1]

    for i in range(0, -1, -1):
        y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

    print("Original y_train:", y_train.sum(axis=0))
    print("Multilabel version:", y_train_multi.sum(axis=0))
    print("Y train ")
    print(y_train_multi)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_multi, test_size=0.15, random_state=42)
    #print(y_train)
    return x_train, x_val, y_train, y_val