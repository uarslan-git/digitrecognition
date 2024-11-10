#import keras
#import os
#import cv2
#import matplotlib.pyplot as plt
#import numpy as np
#
#model = keras.models.load_model(filepath="./digit.keras")
#
##mnist = keras.datasets.mnist
##
##(x_train, y_train), (x_test, y_test) = mnist.load_data()
##
##x_train = keras.utils.normalize(x_train, axis=1)
##x_test = keras.utils.normalize(x_test, axis=1)
##
##loss, accuracy = model.evaluate(x_test, y_test)
#
##print(loss)
##print(accuracy)
#
#digitNum = 1
#
#while os.path.isfile(f"./digits/digit{digitNum}.png"):
#    try:
#        img = cv2.imread(f"./digits/digit{digitNum}.png")[:,:,0]
#        img = np.array([img])
#        prediction = model.predict(img)
#        print(f"This digit is most likely a {np.argmax(prediction)}")
#        plt.imshow(img[0], cmap=plt.cm.binary)
#        plt.show()
#    except:
#        print("Error sir")
#    finally:
#        digitNum += 1

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained model
model = load_model(filepath="./digit.keras")

# Create an activation model to output activations from all layers
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.layers[0].input, outputs=layer_outputs)


# Function to visualize activations
def plot_layer_activations(activations, layer_names):
    for layer_name, activation in zip(layer_names, activations):
        print(f"Visualizing activations for layer: {layer_name}")
        
        # If it's a Conv2D layer
        if len(activation.shape) == 4:  # (batch_size, height, width, channels)
            n_features = activation.shape[-1]  # Number of channels (features)
            size = activation.shape[1]         # Feature map size (height or width)

            # Create a grid to display all the feature maps
            display_grid = np.zeros((size, size * n_features))

            for i in range(n_features):
                feature_map = activation[0, :, :, i]  # Extract the ith feature map
                feature_map -= feature_map.mean()     # Normalize for better visualization
                feature_map /= (feature_map.std() + 1e-5)
                feature_map *= 64
                feature_map += 128
                feature_map = np.clip(feature_map, 0, 255).astype('uint8')

                # Place the feature map in the correct position on the grid
                display_grid[:, i * size:(i + 1) * size] = feature_map

            # Plot the grid of feature maps
            plt.figure(figsize=(20, 15))
            plt.title(f"Activations for {layer_name}")
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()
        
        # If it's a Dense layer
        elif len(activation.shape) == 2:  # (batch_size, neurons)
            plt.figure(figsize=(10, 4))
            plt.title(f"Activations for {layer_name}")
            plt.grid(False)
            plt.plot(activation[0])
            plt.xlabel('Neuron Index')
            plt.ylabel('Activation')
            plt.show()

# Get layer names
layer_names = [layer.name for layer in model.layers]

digitNum = 1

while os.path.isfile(f"./digits/digit{digitNum}.png"):
    try:
        # Load and preprocess the image
        img = cv2.imread(f"./digits/digit{digitNum}.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  # Resize to match the model's input size if needed
        img = img / 255.0  # Normalize to range [0, 1]
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)   # Add batch dimension

        # Get the model's prediction
        prediction = model.predict(img)
        print(f"This digit is most likely a {np.argmax(prediction)}")
        
        # Display the input image
        plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.show()

        # Get activations for the input image
        activations = activation_model.predict(img)

        # Visualize the activations for each layer
        plot_layer_activations(activations, layer_names)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        digitNum += 1

