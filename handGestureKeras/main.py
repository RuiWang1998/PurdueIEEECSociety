from __future__ import print_function
import platform
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from ModelKeras import Conv2Dense2, handCNN
from dataloader import loadData, input_shape
from trainTest import save_model_keras, firstTrain

# this section defines constants
DOWNSCALING_FACTOR = 0.2                           # this means the number of pixels are reduced to downscaling_factor ^ 2 time of its orginal value
TRAIN_FOLDER = 'image_train_folder'               # this is where the processed image goes
TEST_FOLDER = 'image_test_folder'
ALL_FOLDER = 'image_folder_rescaled'
PARENT_FOLDER_NAME = 'image_folder'               # this is the parent folder 
SOURCE_WINDOWS = 'C:/'
SOURCE_LINUX = '/mnt/c/'
SECOND_SOURCE = 'Users/Rui/Documents/GitHub/PurdueIEEECSociety/handGesturePytorch/'
DATA_SOURCE = '../../PurdueIEEEdata/'
IMAGE_DIR = './curves/'  

# this needs to change if the platform is changed
if platform.system() == 'Linux':
    SOURCE = SOURCE_LINUX + SECOND_SOURCE
else:
    SOURCE = SOURCE_WINDOWS + SECOND_SOURCE

# Hyper parameters
EPOCHS = 2
BATCH_SIZE = 5
learning_rate = 0.0001
NUM_CLASS = 5

# Loading data from directories
data_train, data_test, all_test = loadData(batch_size=BATCH_SIZE, down_scaling_factor=DOWNSCALING_FACTOR, source_dir=SOURCE+DATA_SOURCE)

input_shape = input_shape(data_train, data_test, downscaling_factor=DOWNSCALING_FACTOR)


if __name__ == '__main__':
    model = firstTrain(input_shape, data_train, data_test, epochs = 1, model = handCNN)
    save_model_keras(model, dir_json = "model.json", dir_name_weight = './models/first_try.h5')