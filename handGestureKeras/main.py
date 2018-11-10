from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as kbackend
from keras.preprocessing.image import ImageDataGenerator
from ModelKeras import Conv2Dense2
from dataloader import loadData

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
random.seed(1)                                   # this controls the random seed so that the result is reproducible
NUM_CLASS = 5
TEST_PORTION = 0.8
# this needs to change if the platform is changed
SOURCE = SOURCE_WINDOWS + SECOND_SOURCE

# Hyper parameters
EPOCHS = 40
BATCH_SIZE = 30
learning_rate = 0.0001

# Loading data from directories
data_train, data_test = loadData(batch_size = BATCH_SIZE, down_scaling_factor =  DOWNSCALING_FACTOR)

# introducing the model
net1 = Conv2Dense2(input_shape, num_category, lossfunc = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Nadam())

model_log = net1.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))

score = net1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_digit_json = net1.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
net1.save_weights("model_digit.h5")
print("Saved model to disk")