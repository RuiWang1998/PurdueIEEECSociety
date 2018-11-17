from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Lambda, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from ModelKeras import Conv2Dense2, handCNN, handDenseNet
from dataloader import loadData, input_shape, generator, val_generator
from trainTest import save_model_keras, firstTrain, loadAndTrain, loadAndTest
from coupleMaker import euclidean_distance, create_couple, create_wrong, contrastive_loss
from constants import DOWNSCALING_FACTOR, TRAIN_FOLDER, TEST_FOLDER, ALL_FOLDER, PARENT_FOLDER_NAME, SOURCE, EPOCHS, BATCH_SIZE, learning_rate, NUM_CLASS, DATA_SOURCE, DROP_RATE, GROWTH_RATE

# Loading data from directories
# data_train, data_test, all_test = loadData(batch_size=BATCH_SIZE, down_scaling_factor=DOWNSCALING_FACTOR, source_dir=SOURCE+DATA_SOURCE)

input_shape = input_shape(downscaling_factor=DOWNSCALING_FACTOR)

## this is the part for new gesture input, this part is largely inspired by https://github.com/normandipalo/faceID_beta/blob/master/faceid_beta.ipynb
im_in = Input(shape=input_shape)
# model = handCNN(input_shape, NUM_CLASS)
model = handDenseNet(growth_rate = GROWTH_RATE, lastLinear = 100, num_out=200)
x1 = model(im_in)
x1 = BatchNormalization()(x1)
feat_x = Dense(128, activation="linear")(x1)
feat_x = Lambda(lambda  x: K.l2_normalize(x, axis=1))(feat_x)

model_top = Model(inputs = [im_in], outputs = feat_x)

im_in1 = Input(shape=input_shape)
im_in2 = Input(shape=input_shape)

feat_x1 = model_top(im_in1)
feat_x2 = model_top(im_in2)

lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])
model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)

model_final.summary()

model_final.compile(optimizer = keras.optimizers.Adam(), loss = contrastive_loss)

# Here we create a model that outputs the embedding of an input face instead of the distance between two embeddings, so we can map those outputs.
im_in1 = Input(shape=(input_shape))
#im_in2 = Input(shape=(200,200,4))

feat_x1 = model_top(im_in1)
#feat_x2 = model_top(im_in2)

model_output = Model(inputs = im_in1, outputs = feat_x1)

model_output.summary()

model_output.compile(optimizer = keras.optimizers.Adam(lr = 0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True), loss = contrastive_loss)

# model.compile(loss=loss, optimizer=optimizer)
# model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(lr = 0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True), metrics=['accuracy'])


if __name__ == '__main__':

    # this is the first train
    weights_file = './models/denseID.h5'
    json_file_name = "denseID.json"

    model = firstTrain(input_shape, generator(BATCH_SIZE, SOURCE+DATA_SOURCE+TRAIN_FOLDER), val_generator(1, SOURCE+DATA_SOURCE+TEST_FOLDER), epochs = EPOCHS, dir_name_weight = weights_file, dir_json = json_file_name, model = model_final)
    cop = create_couple(SOURCE + DATA_SOURCE)
    model.evaluate([cop[0].reshape((1, input_shapes[0], input_shapes[1], input_shapes[2])), cop[1].reshape((1, input_shapes[0], input_shapes[1], input_shapes[2]))], np.array([0.]))

   #  model = loadAndTrain(data_train, all_test, weights_file, epoch = int(EPOCHS * 5), json_name = json_file_name)
    # save_model_keras(model, dir_json = json_file_name, dir_name_weight = weights_file)
    # loadAndTest(all_test, weights_file, json_file_name)