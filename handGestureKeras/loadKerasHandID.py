from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Lambda, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from ModelKeras import Conv2Dense2, handCNN, handDenseNet
from dataloader import loadData, input_shape
from trainTest import save_model_keras, firstTrain, loadAndTrain, loadAndTest, load_model_keras
from coupleMaker import euclidean_distance, vanilla_loss, soft_max_crossEntropy, contrastive_loss
from constants import DOWNSCALING_FACTOR, TRAIN_FOLDER, TEST_FOLDER, ALL_FOLDER, PARENT_FOLDER_NAME, SOURCE, EPOCHS, BATCH_SIZE, learning_rate, NUM_CLASS, DATA_SOURCE, DROP_RATE, GROWTH_RATE, VIS_FOLDER, VIS_SMALLER
from visualization import PCA_image, PCA_out, get_outputs, save_mean, create_input

# Loading data from directories
data_train, data_test, all_test = loadData(batch_size=BATCH_SIZE, down_scaling_factor=DOWNSCALING_FACTOR, source_dir=SOURCE+DATA_SOURCE)

input_shapes = input_shape()
model = handDenseNet(growth_rate = GROWTH_RATE, lastLinear = 100, num_out=100)
# model = Conv2Dense2()

model.summary()
model.compile(loss = contrastive_loss, optimizer = keras.optimizers.Adam(lr = learning_rate, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True))


if __name__ == '__main__':

    # this is the first train
    weights_file = './models/denseHandID.h5'
    json_file_name = "denseHandID.json"
    model_name = 'denseHandID'
    

    model = firstTrain(input_shape, data_train, data_test, epochs = int(EPOCHS * 0.05), dir_name_weight = weights_file, dir_json = json_file_name, model = model)

    # model = load_model_keras(json_file_name, weights_file)
    # model.compile(loss = vanilla_loss, optimizer =  keras.optimizers.Adam(lr = learning_rate, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True))

    #cop = create_input(SOURCE+DATA_SOURCE+VIS_FOLDER+"/4/bowman_two_fingers0.jpg", factor=DOWNSCALING_FACTOR)
    #print("\n\n")
    #print(model.predict(cop.reshape((1,input_shapes[0],input_shapes[1],input_shapes[2]))))
    #print("\n\n")

    #outputs, mean = get_outputs(model, SOURCE+DATA_SOURCE+VIS_FOLDER, dim = 5, factor=DOWNSCALING_FACTOR)
    #save_mean(outputs, model_name+'all')
    #save_mean(mean[:,0,:], model_name)
    #PCA_image(PCA_out(outputs), name = 'after', item = 196)
    # model = loadAndTrain(data_train, all_test, weights_file, epoch = int(EPOCHS * 1), json_name = json_file_name)
    # save_model_keras(model, dir_json = json_file_name, dir_name_weight = weights_file)
    # loadAndTest(all_test, weights_file, json_file_name)