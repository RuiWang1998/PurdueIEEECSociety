import platform
import random

# this section defines constants
DOWNSCALING_FACTOR = 0.1                         # this means the number of pixels are reduced to downscaling_factor ^ 2 time of its orginal value
TRAIN_FOLDER = 'image_train_folder'               # this is where the processed image goes
TEST_FOLDER = 'image_test_folder'
ALL_FOLDER = 'image_folder_rescaled'
TRAIN_AUG = 'image_train_augmented'
TEST_AUG = 'image_test_augmented'
PARENT_FOLDER_NAME = 'image_folder'               # this is the parent folder 
SOURCE_WINDOWS = 'C:/'
SOURCE_LINUX = '/mnt/c/'
SECOND_SOURCE = 'Users/Rui/Documents/GitHub/PurdueIEEECSociety/handGestureKeras/'
DATA_SOURCE = '../../PurdueIEEEdata/'
IMAGE_DIR = './curves/'  
DROP_RATE = 0.5
VIS_FOLDER = 'image_visual_folder'
VIS_SMALLER = 'image_visual_smaller'

# this needs to change if the platform is changed
if platform.system() == 'Linux':
    SOURCE = SOURCE_LINUX + SECOND_SOURCE
else:
    SOURCE = SOURCE_WINDOWS + SECOND_SOURCE

# Hyper parameters
EPOCHS = 100
BATCH_SIZE = 10
learning_rate = 0.0001
NUM_CLASS = 5

input_channel = 3
resolution = (480, 640)
GROWTH_RATE = 15
TEST_PORTION = 0.8

random.seed(2)

MEAN_PERCENT = 0.8