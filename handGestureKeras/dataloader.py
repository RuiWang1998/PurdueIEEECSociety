from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as kbackend
import numpy as np

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
seed = 1
NUM_CLASS = 5
TEST_PORTION = 0.8
# this needs to change if the platform is changed
SOURCE = SOURCE_WINDOWS + SECOND_SOURCE + THIRD_SOURCE

# Hyper parameters
BATCH_SIZE = 30
learning_rate = 0.0001
input_channel = 3

resolution = np.array([480, 640])

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        channel_shift_range = 0.4,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

def loadData(batch_size = BATCH_SIZE, down_scaling_factor =  DOWNSCALING_FACTOR):
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            SOURCE + TRAIN_FOLDER,  # this is the target directory
            target_size=resolution * down_scaling_factor,  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            SOURCE + TEST_FOLDER,
            target_size=resolution * down_scaling_factor,
            batch_size=batch_size,
            class_mode='categorical')

    return train_generator, validation_generator

def input_shape(train_data, test_data):
    # Keras can work with datasets that have their channels as the first dimension ('channels_first') or 'channels_last'
    if kbackend.image_data_format() == 'channels_first':
        input_shape = np.array([input_channel, resolution[0], resolution[1]])
    else:
        input_shape = np.array([resolution[0], resolution[1], input_channel])

    return input_shape