import sys

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
	return np.exp(x)/( 1. + np.exp(x) )


# 1. Prepare data: marks vs hours of studying (reviewing)
X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

if(True):
	# [array([[1.5909579]], dtype=float32), array([-4.091234], dtype=float32)]
	plt.plot(X, 1.59*X - 4.09, 'r-')
	# plt.plot(X, sigmoid(X), 'b--')
	plt.plot(X,y)
	plt.show()
	sys.exit()


# 2. Build model
xmodel = keras.models.Sequential()
xmodel.add( keras.layers.Dense( 1, input_shape=(1,) ) )
xmodel.add( keras.layers.Activation('sigmoid') )

# 3. Gradient descent optimizer and loss function
sgd = keras.optimizers.SGD(lr=0.05)
xmodel.compile( loss=keras.losses.binary_crossentropy,
                optimizer=sgd )

# 4. Train the model
xmodel.fit( X, y, epochs=3000, batch_size=1 )

res = xmodel.get_weights()

print( res )