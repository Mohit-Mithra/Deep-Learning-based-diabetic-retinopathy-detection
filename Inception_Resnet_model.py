from keras.applications import inception_resnet_v2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import layers
from keras.optimizers import Adam

def build_model_Inception_Resnet():

    inception_resnet = inception_resnet_v2.InceptionResNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
    )

    model = Sequential()
    model.add(inception_resnet)
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(2, activation='sigmoid'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.0005),
        metrics=['accuracy']
    )
    
    return model
