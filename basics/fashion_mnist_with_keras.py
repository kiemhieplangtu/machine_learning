import sys

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def display_images(images, class_names, labels):
    plt.figure(figsize=(10,10))
    grid_size = min(25, len(images))
    for i in range(grid_size):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])



## --- MAIN --- ##

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print('x_train shape: \t', x_train.shape)
print('y_train shape: \t', y_train.shape)
print('x_test shape: \t', x_test.shape)
print('y_test shape: \t', y_test.shape)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(y_train)


if(False):
	plt.figure()
	plt.imshow(x_train[1])
	plt.colorbar()
	plt.grid(False)
	plt.show()
	sys.exit()

# nomalize data
x_train = x_train/255.
x_test  = x_test/255.

num_classes = 10

# convert class vectors to binary class matrices: on-hot encoding
y_train = keras.utils.to_categorical( y_train, num_classes )
y_test  = keras.utils.to_categorical( y_test, num_classes )

# 2. Build model
xmodel = keras.models.Sequential()
xmodel.add( keras.layers.Flatten( input_shape=(28,28) ) )
xmodel.add( keras.layers.Dense( 128, activation='relu') )
xmodel.add( keras.layers.Dense( 256, activation='relu') )
xmodel.add( keras.layers.Dense( 512, activation='relu') )

xmodel.add( keras.layers.Dense( num_classes, activation='softmax' ))

# 3. Loss, optimizer and metrics
xmodel.compile( loss= keras.losses.categorical_crossentropy,
	            optimizer=keras.optimizers.SGD(learning_rate=0.1),
	            metrics=['accuracy'])




xmodel.fit(x_train, y_train, epochs=20)

res = xmodel.evaluate( x_test, y_test, verbose=False )
print( 'Test loss: %.4f' % res[0] )
print( 'Test accuracy: %.4f' % res[1] )


prd = xmodel.predict(x_test)


print(prd)


display_images(x_test, class_names, np.argmax(prd, axis = 1))
plt.show()