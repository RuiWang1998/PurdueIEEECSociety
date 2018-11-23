import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, merge, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, SeparableConv2D, Subtract, LeakyReLU
from keras.activations import relu, softmax
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K

from constants import DROP_RATE
from dataloader import input_shape
from coupleMaker import contrastive_loss

# Creating the Model
def Conv2Dense2(num_category = 5):
    Conv2Dense2 = Sequential()
    Conv2Dense2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape()))   # 32 - 3 X 3 filters with ReLU Activation Function
    Conv2Dense2.add(Conv2D(64, (3, 3), activation='relu'))   # 64 - 3 X 3 filters with ReLU Activation Function
    Conv2Dense2.add(MaxPooling2D(pool_size=(2, 2)))   # Max Pool the bitmaps 4 bit squares at a time
    Conv2Dense2.add(Flatten())                        # Flatten the dimensions
    Conv2Dense2.add(Dense(128, activation='relu'))    # Adding a dense layer at the end
    Conv2Dense2.add(Dense(num_category, activation='softmax'))   # Softmax activation function to get probability distributions

    return Conv2Dense2

def handCNN(num_category = 5):
    handCNN = Sequential()
    # the first convolutional layer
    handCNN.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape(), strides=(1,1)))
    handCNN.add(BatchNormalization())
    handCNN.add(Activation('relu'))
    handCNN.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

    # the second convolutional layer
    handCNN.add(Conv2D(32, kernel_size=(5, 5), input_shape=input_shape(), strides=(1,1)))
    handCNN.add(BatchNormalization())
    handCNN.add(Activation('relu'))
    handCNN.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # the third convolutional layer
    handCNN.add(Conv2D(32, kernel_size=(6, 6), input_shape=input_shape(), strides=(1,1)))
    handCNN.add(BatchNormalization())
    handCNN.add(Activation('relu'))
    handCNN.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1))) # for recognition only, this one is 2

    # the fourth convolutional layer
    handCNN.add(Conv2D(32, kernel_size=(9, 9), input_shape=input_shape(), strides=(1,1)))
    handCNN.add(BatchNormalization())
    handCNN.add(Activation('relu'))
    handCNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # flatten the dimensions for the linear layers
    handCNN.add(Flatten())
    handCNN.add(Dense(32, activation = 'sigmoid'))
    handCNN.add(Dense(num_category, activation='relu'))

    return handCNN

def single_layer(x, growth_rate, kernel_size, stride1, pool_size, stride2, i, activation = 'relu'):

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(i * growth_rate, kernel_size=1, strides=1, padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters=growth_rate, kernel_size = kernel_size, strides=stride1, padding='same')(x)

    return x

def denseBlock(x, growth_rate, kernel_sizes, stride1, stride2, activation = 'relu', layer_num = 3):

    for i, kernel_size in enumerate(kernel_sizes):
        x1 = single_layer(x, growth_rate, kernel_size = kernel_size, stride1 = 1,  pool_size = 3, stride2 = 1, i = i + 1, activation = 'relu')
        if i != len(kernel_sizes) - 1:
            x = concatenate([x1, x], axis=3)

    return x1

def transition_layer(x, growth_rate, pool_size = 2, strides = 2, activation = LeakyReLU(alpha = 0.3)):
    
    x = Conv2D(growth_rate, kernel_size = 2, strides=1, padding='same')(x)
    x = activation(x)
    x = AveragePooling2D(pool_size = pool_size, strides = 2)(x)

    return x

def handDenseNet(input_shapes = input_shape(), growth_rate = 10, num_out = 5, optimizer = keras.optimizers.Adam(lr = 0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True), loss=keras.losses.categorical_crossentropy, lastLinear = 30):

    img_input=Input(shape=input_shapes)
    # this is the input layer
    x = Conv2D(growth_rate, (2, 2), strides=(1, 1), padding='same')(img_input)
    x = MaxPooling2D(pool_size=2, strides=1, padding='same')(x)

    x = denseBlock(x, growth_rate, kernel_sizes = (2, 4, 6), stride1 = 1, stride2 = 1, activation = 'relu', layer_num=3)

    x = transition_layer(x, growth_rate, pool_size = 2, strides = 2)

    x = denseBlock(x, growth_rate, kernel_sizes = (2, 3, 4), stride1 = 1, stride2 = 1, activation = 'relu', layer_num=3)

    x = transition_layer(x, growth_rate, pool_size = 2, strides = 2)

    x = denseBlock(x, growth_rate, kernel_sizes = (1, 2, 3), stride1 = 1, stride2 = 1, activation = 'relu', layer_num=3)

    x = Flatten()(x)
    x = Dense(lastLinear, activation = 'relu')(x)
    x = Dense(num_out)(x)
    x = LeakyReLU(alpha=0.3)(x)

    net = Model(img_input, x)

    return net
