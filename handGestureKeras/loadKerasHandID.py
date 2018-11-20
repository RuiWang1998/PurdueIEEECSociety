from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Lambda, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from ModelKeras import Conv2Dense2, handCNN, handDenseNet
from dataloader import loadData, input_shape
from trainTest import save_model_keras, firstTrain, loadAndTrain, loadAndTest
from coupleMaker import euclidean_distance, 
from constants import DOWNSCALING_FACTOR, TRAIN_FOLDER, TEST_FOLDER, ALL_FOLDER, PARENT_FOLDER_NAME, SOURCE, EPOCHS, BATCH_SIZE, learning_rate, NUM_CLASS, DATA_SOURCE, DROP_RATE, GROWTH_RATE

# Loading data from directories
data_train, data_test, all_test = loadData(batch_size=BATCH_SIZE, down_scaling_factor=DOWNSCALING_FACTOR, source_dir=SOURCE+DATA_SOURCE)

input_shape = input_shape()
model = handDenseNet(growth_rate = GROWTH_RATE, lastLinear = 100)
# model = Conv2Dense2()

model.summary()
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(lr = learning_rate, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True), metrics=['accuracy'])


if __name__ == '__main__':

    # this is the first train
    weights_file = './models/denseBiggerBest.h5'
    json_file_name = "modelDenseBigger.json"

    # model = firstTrain(input_shape, data_train, data_test, epochs = EPOCHS, dir_name_weight = weights_file, dir_json = json_file_name, model = model)
    model.predict([cop[0].reshape((1,input_shapes[0],input_shapes[1],input_shapes[2])), cop[1].reshape((1,input_shapes[0],input_shapes[1],input_shapes[2]))])

    model = loadAndTrain(data_train, all_test, weights_file, epoch = int(EPOCHS * 1), json_name = json_file_name)
    # save_model_keras(model, dir_json = json_file_name, dir_name_weight = weights_file)
    # loadAndTest(all_test, weights_file, json_file_name)