import sys

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# 1. prepare data y = 2*x0 + 3*x1 + 4 with noise
y_noise = 0.2*np.random.randn(100)
X       = np.random.rand(100, 2)
y       = 2.*X[:,0] + 3.*X[:,1] + 4. + y_noise

# Build model
xmodel = keras.models.Sequential()
xmodel.add( keras.layers.Dense(1, input_shape=(2,)) )
xmodel.add( keras.layers.Activation('linear') )

# 3. Gradient descent optimizer and loss function
sgd = keras.optimizers.SGD(lr=0.1)
xmodel.compile( loss='mse', optimizer=sgd )
xmodel.fit( X, y, epochs=100, batch_size=2 )
res = xmodel.get_weights()

print(res)