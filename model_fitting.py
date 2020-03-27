import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
from random import shuffle
import pickle 
from tensorflow import keras
import matplotlib.pyplot as plt


X = np.load("X.npy")
y = np.load("y.npy")


# Building the model
model = keras.models.Sequential()

# 3 convolutional layers
model.add(keras.layers.Conv2D(32, (3, 3), input_shape = X.shape[1:])
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))


# 2 hidden layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128))
model.add(keras.layers.Activation("relu"))

model.add(keras.layers.Dense(128))
model.add(keras.layers.Activation("relu"))

# The output layer with 10neurons, for 10 classes
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation("softmax"))

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])


model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)

model.save('my_model.h5')

#INCEPTION_V3
	  
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD

base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=features.shape[1:])

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
	  
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True
	  
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(features,pd.get_dummies(labels),batch_size = 32, epochs=40, validation_split=0.2,shuffle = True)	  

model.save('Inception_v3.h5')
	  
# Printing a graph showing the accuracy changes during the training phase
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
