import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_train, axis=1)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))

model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train, epochs=25)

model.save('digit.keras')
