from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as kbackend
from keras.preprocessing.image import ImageDataGenerator
from ModelKeras import Conv2Dense2


# Loading data from the MNIST Database
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train shape", X_train.shape)   # Shape should be 60000 X 28 X 28
print("y_train shape", y_train.shape)   # Shape should be 60000 X 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTE: Change this when working with other images for the project
img_rows, img_cols = 28, 28
num_category = 10
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Keras can work with datasets that have their channels as the first dimension ('channels_first') or 'channels_last'
if kbackend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)   # 1 is used here because MNIST is B&W
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Convert datatypes of the numpy arrays to 32 bit floats and divide by 255 (batch normalization) Why is this BN?
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)     # Shape should be 60000 X 28 X 28 X 1
print(X_train.shape[0], 'train samples')   # 60000 Training Samples
print(X_test.shape[0], 'test samples')     # 10000 Testing Samples

# Convert the labels to one hot form
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

# introducing the model
net1 = Conv2Dense2(input_shape, num_category, lossfunc = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Nadam())

# Training Hyperparameters
batch_size = 100   # Mini batch sizes
num_epoch = 10     # Number of epochs to train for
model_log = net1.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))

score = net1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_digit_json = net1.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
net1.save_weights("model_digit.h5")
print("Saved model to disk")