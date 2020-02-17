import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('train.csv')
N = train_df.shape[0]
x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)
for i, image_id in enumerate((train_df['id_code'])):
     

    x_train[i, :, :, :] = Image.open('processed_train_images/'+image_id+'.png')

train_df['diagnosis'] = train_df['diagnosis'] > 0
train_df['diagnosis'] = train_df['diagnosis'] * 1.0

y_train = pd.get_dummies(train_df['diagnosis']).values

print(x_train.shape)
print(y_train.shape)

y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 1] = y_train[:, 1]

for i in range(0, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_multi, test_size=0.15, random_state=42)

print(y_train)
