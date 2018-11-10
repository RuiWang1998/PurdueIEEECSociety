from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from ModelKeras import Conv2Dense2
from dataloader import loadData, input_shape


# this section defines constants
DOWNSCALING_FACTOR = 0.2                           # this means the number of pixels are reduced to downscaling_factor ^ 2 time of its orginal value
TRAIN_FOLDER = 'image_train_folder'               # this is where the processed image goes
TEST_FOLDER = 'image_test_folder'
ALL_FOLDER = 'image_folder'
PARENT_FOLDER_NAME = 'image_folder'               # this is the parent folder 
SOURCE_WINDOWS = 'C:/'
SOURCE_LINUX = '/mnt/c/'
SECOND_SOURCE = 'Users/Rui/Documents/GitHub/PurdueIEEECSociety/handGesturePytorch/'
THIRD_SOURCE = '../../PurdueIEEEdata/'
IMAGE_DIR = './curves/'      
# this needs to change if the platform is changed
SOURCE = SOURCE_WINDOWS + SECOND_SOURCE

# Hyper parameters
EPOCHS = 1
BATCH_SIZE = 30
learning_rate = 0.0001
NUM_CLASS = 5

# Loading data from directories
data_train, data_test = loadData(batch_size=BATCH_SIZE, down_scaling_factor=DOWNSCALING_FACTOR)

input_shape = input_shape(data_train, data_test, downscaling_factor=DOWNSCALING_FACTOR)
# introducing the model
net1 = Conv2Dense2(input_shape, NUM_CLASS, lossfunc = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Nadam(lr = learning_rate))

validation_step = len(data_test)
train_step = len(data_train)

net1.fit_generator(
        generator=data_train,
        steps_per_epoch = train_step,
        epochs=1,
        validation_data=data_test,
        validation_steps=train_step)

net1.save_weights('first_try.h5')

print("Saved model to disk")