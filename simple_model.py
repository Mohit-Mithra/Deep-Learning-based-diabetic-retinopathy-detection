from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import layers
from keras.optimizers import Adam

def build_simple_model():

    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same',input_shape=(224,224,3)))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    #model.add(layers.Dropout(0.5))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']
    )
    
    return model
