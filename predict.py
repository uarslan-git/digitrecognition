import keras
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model(filepath="./digit.keras")

#mnist = keras.datasets.mnist
#
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#x_train = keras.utils.normalize(x_train, axis=1)
#x_test = keras.utils.normalize(x_test, axis=1)
#
#loss, accuracy = model.evaluate(x_test, y_test)

#print(loss)
#print(accuracy)

digitNum = 1

while os.path.isfile(f"./digits/digit{digitNum}.png"):
    try:
        img = cv2.imread(f"./digits/digit{digitNum}.png")[:,:,0]
        img = np.array([img])
        prediction = model.predict(img)
        print(f"This digit is most likely a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error sir")
    finally:
        digitNum += 1
