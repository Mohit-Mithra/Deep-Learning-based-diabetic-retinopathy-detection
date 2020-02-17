from keras.applications import DenseNet121
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import layers
from keras.optimizers import Adam

def build_model():

    densenet = DenseNet121(
    weights='DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
    )

    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.0001),
        metrics=['accuracy']
    )
    
    return model
