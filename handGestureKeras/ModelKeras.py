import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, merge
from constants import DROP_RATE

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

#def Dense():
    
#    inputs = Input(shape=in_shp)
#    x=Reshape([1]+in_shp)(inputs)
#    #conv 1
#    #x=ZeroPadding2D((0, 1))(x)
#    x=Conv2D(256, 1, 3, border_mode="same",data_format='channels_first', activation="sigmoid", name="conv1", init='glorot_uniform')(x)
#    x=Dropout(DROP_RATE)(x)
#    list_feat=[x]

#    #conv2
#    #x=ZeroPadding2D((0, 1))(x)
#    x=Conv2D(256, 2, 3, border_mode="same",data_format='channels_first', activation="sigmoid", name="conv2", init='glorot_uniform')(x)
#    x=Dropout(DROP_RATE)(x)
#    list_feat.append(x)
#    m1 = merge(list_feat, mode='concat', concat_axis=1)

#    #conv 3
#    #x=ZeroPadding2D((0, 1))(x)
#    x=Conv2D(80, 1, 3, border_mode="same",data_format='channels_first', activation="sigmoid", name="conv3", init='glorot_uniform')(m1)
#    x=Dropout(DROP_RATE)(x)
#    list_feat.append(x)
#    m2=merge(list_feat, mode='concat', concat_axis=1)

#    # conv 4
#    x=Conv2D(80, 1, 3, border_mode="same",data_format='channels_first', activation="sigmoid", name="conv4", init='glorot_uniform')(m2)
#    x=Dropout(DROP_RATE)(x)
#    list_feat.append(x)
#    m3=merge(list_feat, mode='concat', concat_axis=1)

#    x=Flatten()(m3)
#    # Linear Layer 1
#    x=Dense(256, activation='relu', init='he_normal', name="Linear 1")(x)
#    x=Dropout(DROP_RATE)(x)
#    # Linear Layer 2
#    x=Dense(len(classes), init='he_normal', name="dense2" )(x)
#    model = Model(input=[inputs], output=[x])

#    return model