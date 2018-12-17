# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


print(train_images.shape)
# => (60000, 28, 28)
print("train_labels", train_labels.shape)
# => train_labels (60000,)
print( train_labels)

print(test_images.shape)
#  => (10000, 28, 28)


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()