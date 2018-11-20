import platform
import torch
import torch.optim as optim
import torch.nn as nn
import glob
import os

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# this section defines constants
DOWNSCALING_FACTOR = 0.2                          # this means the number of pixels are reduced to downscaling_factor ^ 2 time of its orginal value
TRAIN_FOLDER = 'image_train_folder'               # this is where the processed image goes
TEST_FOLDER = 'image_test_folder'
ALL_FOLDER = 'image_folder_rescaled'
TRAIN_AUG = 'image_train_augmented'
TEST_AUG = 'image_test_augmented'
PARENT_FOLDER_NAME = 'image_folder'               # this is the parent folder 
SOURCE_WINDOWS = 'C:/'
SOURCE_LINUX = '/mnt/c/'
SECOND_SOURCE = 'Users/Rui/Documents/GitHub/PurdueIEEECSociety/handGesturePytorch/'
DATA_SOURCE = '../../PurdueIEEEdata/'
IMAGE_DIR = './curves/'  
VIS_FOLDER = 'image_visual_folder'

# this needs to change if the platform is changed
if platform.system() == 'Linux':
    SOURCE = SOURCE_LINUX + SECOND_SOURCE
else:
    SOURCE = SOURCE_WINDOWS + SECOND_SOURCE

# Hyper parameters
EPOCHS = 7
BATCH_SIZE = 30
learning_rate = 0.00001
NUM_CLASS = 5
GROWTH_RATE = 15
TEST_PORTION = 0.8
resolution = (480, 640)

def optimizer(model, adam = True):
    if adam:
        theOptimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.99, 0.999), 
                           eps=1e-08, weight_decay=0, amsgrad=True)
    return theOptimizer

loss_func = nn.CrossEntropyLoss()       # I am only using cross entropy loss for now, I would love to add Wasserstein distance into the loss function later on to smooth the update

train_count = 0
for folderName in glob.glob(SOURCE+DATA_SOURCE+TRAIN_FOLDER + '/*'):
    path, dirs, files = next(os.walk(folderName +"/"))
    train_count += len(files)

test_count = 0
for folderName in glob.glob(SOURCE+DATA_SOURCE+TEST_FOLDER + '/*'):
    path, dirs, files = next(os.walk(folderName +"/"))
    test_count += len(files)