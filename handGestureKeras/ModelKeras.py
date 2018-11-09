import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Creating the Model
def Conv2Dense2(input_shape, num_category, lossfunc = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta()):
    Conv2Dense2 = Sequential()
    Conv2Dense2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))   # 32 - 3 X 3 filters with ReLU Activation Function
    Conv2Dense2.add(Conv2D(64, (3, 3), activation='relu'))   # 64 - 3 X 3 filters with ReLU Activation Function
    Conv2Dense2.add(MaxPooling2D(pool_size=(2, 2)))   # Max Pool the bitmaps 4 bit squares at a time
    Conv2Dense2.add(Flatten())                        # Flatten the dimensions
    Conv2Dense2.add(Dense(128, activation='relu'))    # Adding a dense layer at the end
    Conv2Dense2.add(Dense(num_category, activation='softmax'))   # Softmax activation function to get probability distributions
    # Categorical Crossentropy loss function with Adadelta optimizer
    Conv2Dense2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    return Conv2Dense2
