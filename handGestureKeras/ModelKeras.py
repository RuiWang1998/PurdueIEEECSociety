import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation

# Creating the Model
def Conv2Dense2(input_shape, num_category, lossfunc = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(lr = 0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True)):
    Conv2Dense2 = Sequential()
    Conv2Dense2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))   # 32 - 3 X 3 filters with ReLU Activation Function
    Conv2Dense2.add(Conv2D(64, (3, 3), activation='relu'))   # 64 - 3 X 3 filters with ReLU Activation Function
    Conv2Dense2.add(MaxPooling2D(pool_size=(2, 2)))   # Max Pool the bitmaps 4 bit squares at a time
    Conv2Dense2.add(Flatten())                        # Flatten the dimensions
    Conv2Dense2.add(Dense(128, activation='relu'))    # Adding a dense layer at the end
    Conv2Dense2.add(Dense(num_category, activation='softmax'))   # Softmax activation function to get probability distributions
    # Categorical Crossentropy loss function with Adadelta optimizer
    Conv2Dense2.compile(loss=lossfunc, optimizer=optimizer, metrics=['accuracy'])

    return Conv2Dense2

def handCNN(input_shape, num_category, lossfunc = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(lr = 0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, amsgrad=True)):
    handCNN = Sequential()
    # the first convolutional layer
    handCNN.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, strides=(1,1)))
    handCNN.add(BatchNormalization())
    handCNN.add(Activation('relu'))
    handCNN.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

    # the first convolutional layer
    handCNN.add(Conv2D(32, kernel_size=(5, 5), input_shape=input_shape, strides=(1,1)))
    handCNN.add(BatchNormalization())
    handCNN.add(Activation('relu'))
    handCNN.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # the first convolutional layer
    handCNN.add(Conv2D(32, kernel_size=(6, 6), input_shape=input_shape, strides=(1,1)))
    handCNN.add(BatchNormalization())
    handCNN.add(Activation('relu'))
    handCNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # the first convolutional layer
    handCNN.add(Conv2D(32, kernel_size=(9, 9), input_shape=input_shape, strides=(1,1)))
    handCNN.add(BatchNormalization())
    handCNN.add(Activation('relu'))
    handCNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # flatten the dimensions for the linear layers
    handCNN.add(Flatten())
    handCNN.add(Dense(32, activation = 'sigmoid'))
    handCNN.add(Dense(num_category, activation='softmax'))

    handCNN.compile(loss=lossfunc, optimizer=optimizer, metrics=['accuracy'])

    return handCNN