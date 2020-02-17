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


def preprocess_test_data(test_df):
    N = test_df.shape[0]
    x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

    j = 0
    #for i, image_id in enumerate((test_df['id_code'])):
         
        #f(j%500 == 0):
           #print(j)

        #img = preprocess_image('AnotherTest/TestingSet/'+image_id+'.jpg')
        ##img_array = img_to_array(img)
        ##save_img('processed_test_images/'+image_id, img_array)

        #img.save('AnotherTest/TestingSet/'+image_id+'.jpg')

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
    

    for i, image_id in enumerate((test_df['id_code'])):
         

        x_test[i, :, :, :] = Image.open('AnotherTest/TestingSet/'+image_id+'.jpg')

    test_df['diagnosis'] = test_df['diagnosis'] > 0
    test_df['diagnosis'] = test_df['diagnosis'] * 1.0
    # print(x_train.shape)
    # print(y_train.shape)
    # #print(x_test.shape)

    y_test = pd.get_dummies(test_df['diagnosis']).values

    print("y_test shape")
    print(y_test.shape)

    y_test_multi = np.empty(y_test.shape, dtype=y_test.dtype)
    y_test_multi[:, 1] = y_test[:, 1]

    for i in range(0, -1, -1):
         y_test_multi[:, i] = np.logical_or(y_test[:, i], y_test_multi[:, i+1])

    print("Original y_test:", y_test.sum(axis=0))
    print("Multilabel version:", y_test_multi.sum(axis=0))

    return x_test, y_test_multi



