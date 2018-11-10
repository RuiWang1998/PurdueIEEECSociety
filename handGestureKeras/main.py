from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from ModelKeras import Conv2Dense2, handCNN
from dataloader import loadData, input_shape
from trainTest import save_model_keras, firstTrain, loadAndTrain
from constants import DOWNSCALING_FACTOR, TRAIN_FOLDER, TEST_FOLDER, ALL_FOLDER, PARENT_FOLDER_NAME, SOURCE, EPOCHS, BATCH_SIZE, learning_rate, NUM_CLASS, DATA_SOURCE

# Loading data from directories
data_train, data_test, all_test = loadData(batch_size=BATCH_SIZE, down_scaling_factor=DOWNSCALING_FACTOR, source_dir=SOURCE+DATA_SOURCE)

input_shape = input_shape(data_train, data_test, downscaling_factor=DOWNSCALING_FACTOR)

if __name__ == '__main__':
    # this is the first train
    weights_file = './models/first_try.h5'
    json_file_name = "model.json"

    # model = firstTrain(input_shape, data_train, data_test, epochs = 1, model = handCNN)
    # save_model_keras(model, dir_json = json_file_name, dir_name_weight = weights_file)

    model = loadAndTrain(data_train, data_test, weights_file, epoch = 30, json_name = json_file_name)
    save_model_keras(model, dir_json = json_file_name, dir_name_weight = weights_file)