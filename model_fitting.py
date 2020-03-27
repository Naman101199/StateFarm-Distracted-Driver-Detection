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
model.add(keras.layers.Conv2D(32, (3, 3), input_shape = (224,224,3))
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


history = model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)

model.save('my_model.h5')

# Printing a graph showing the accuracy changes during the training phase
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
