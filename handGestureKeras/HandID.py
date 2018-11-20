import keras
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
from matplotlib import image as img

from constants import DOWNSCALING_FACTOR, TRAIN_FOLDER, TEST_FOLDER, ALL_FOLDER, PARENT_FOLDER_NAME, SOURCE, EPOCHS, BATCH_SIZE, learning_rate, NUM_CLASS, DATA_SOURCE, DROP_RATE, GROWTH_RATE, VIS_FOLDER
from dataloader import input_shapes
from imageProcessing import process_image
from ModelKeras import handDenseNet

train_source = SOURCE + DATA_SOURCE + TRAIN_FOLDER
test_source = SOURCE + DATA_SOURCE + TEST_FOLDER
visual_source = SOURCE + DATA_SOURCE + VIS_FOLDER

labels = ["fist", "oneFinger", "openHand", "thumpsUp", "twoFinger"]

## Input preprocessing
# Here we create some functions that will create the input couple for our model, both correct and wrong couples. I created functions to have both depth-only input and RGBD inputs.

def create_couple(file_path):
    folder=np.random.choice(glob.glob(file_path + "/*"))
    while folder == "datalab":
      folder=np.random.choice(glob.glob(file_path + "/*"))
  #  print(folder)
    photo_file = np.random.choice(glob.glob(folder + "/*.jpg"))
    mat1 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    #plt.imshow(mat1)
    #plt.show()

    photo_file = np.random.choice(glob.glob(folder + "/*.jpg"))
    mat2 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    #plt.imshow(mat2)
    #plt.show()
    return np.array([mat1, mat2])

def create_wrong(file_path):
    folder=np.random.choice(glob.glob(file_path + "/*"))
    while folder == "datalab":
      folder=np.random.choice(glob.glob(file_path + "/*"))    
    photo_file = np.random.choice(glob.glob(folder + "/*.jpg"))
    mat1 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    #plt.imshow(mat1)
    #plt.show()
    
    folder2=np.random.choice(glob.glob(file_path + "/*"))
    while folder==folder2 or folder2=="datalab": #it activates if it chose the same folder
        folder2=np.random.choice(glob.glob(file_path + "/*"))
    photo_file = np.random.choice(glob.glob(folder2 + "/*.jpg"))
    mat2 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    #plt.imshow(mat2)
    #plt.show()
  
    return np.array([mat1, mat2])

## Network Crafting
# Now we create the network. We first manually create the constrative loss, then we define the network architecture starting from the SqueezeNet architecture, and then using it as a siamese-network for embedding faces into a manifold. (the network for now is very big and could be heavily optimized, but I just wanted to show a proof-of-concept)
def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sum((K.square(u - v)), axis=1, keepdims=True)
        
def contrastive_loss(y_true,y_pred):
    margin=1.
    return K.sqrt(K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.))))
   # return K.mean(K.square(y_pred))

def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1,1), padding='valid')(x)
    x = Activation('relu')(x)
    
    left = Convolution2D(expand, (1,1), padding='valid')(x)
    left = Activation('relu')(left)
    
    right = Convolution2D(expand, (3,3), padding='same')(x)
    right = Activation('relu')(right)
    
    x = concatenate([left, right], axis=3)
    return x

img_input=Input(shape=(input_shapes[0],input_shapes[1],input_shapes[2]))

x = Convolution2D(64, (5, 5), strides=(2, 2), padding='valid')(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = fire(x, squeeze=16, expand=16)
x = fire(x, squeeze=16, expand=16)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = fire(x, squeeze=32, expand=32)
x = fire(x, squeeze=32, expand=32)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = fire(x, squeeze=48, expand=48)
x = fire(x, squeeze=48, expand=48)
x = fire(x, squeeze=64, expand=64)
x = fire(x, squeeze=64, expand=64)
x = Dropout(0.2)(x)
x = Convolution2D(512, (1, 1), padding='same')(x)
out = Activation('relu')(x)
modelsqueeze= Model(img_input, out)

modelsqueeze.summary()

im_in = Input(shape=(input_shapes[0],input_shapes[1],input_shapes[2]))
#wrong = Input(shape=(200,200,3))
x1 = modelsqueeze(im_in)

#x = Convolution2D(64, (5, 5), padding='valid', strides =(2,2))(x)

#x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)

"""
x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
x1 = Dropout(0.4)(x1)

x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x1)

x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.4)(x1)

x1 = Convolution2D(64, (1,1), padding='same', activation="relu")(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.4)(x1)
"""

x1 = Flatten()(x1)
x1 = Dense(10000, activation="relu")(x1)
x1 = Dropout(0.2)(x1)
#x1 = BatchNormalization()(x1)
feat_x = Dense(128, activation="linear")(x1)
feat_x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x)
model_top = Model(inputs = [im_in], outputs = feat_x)

model_top.summary()

im_in1 = Input(shape=(input_shapes[0],input_shapes[1],input_shapes[2]))
im_in2 = Input(shape=(input_shapes[0],input_shapes[1],input_shapes[2]))

feat_x1 = model_top(im_in1)
feat_x2 = model_top(im_in2)

lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)

model_final.summary()
adam = Adam(lr=0.000000001)
sgd = SGD(lr=0.001, momentum=0.9)
model_final.compile(optimizer=adam, loss=contrastive_loss)

def generator(batch_size, file_path):
  
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
   #   switch += 1
      if switch:
     #   print("correct")
        X.append(create_couple(file_path).reshape((2,input_shapes[0],input_shapes[1],input_shapes[2])))
        y.append(np.array([0.]))
      else:
     #   print("wrong")
        X.append(create_wrong(file_path).reshape((2,input_shapes[0],input_shapes[1],input_shapes[2])))
        y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    XX1=X[0,:]
    XX2=X[1,:]
    yield [X[:,0],X[:,1]],y

def val_generator(batch_size, file_path):
  
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
      if switch:
        X.append(create_couple(file_path).reshape(2,input_shapes[0],input_shapes[1],input_shapes[2]))
        y.append(np.array([0.]))
      else:
        X.append(create_wrong(file_path).reshape(2,input_shapes[0],input_shapes[1],input_shapes[2]))
        y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    XX1=X[0,:]
    XX2=X[1,:]
    yield [X[:,0],X[:,1]],y
    
def save_model_keras(model, dir_json = "faceid.json", dir_name_weight = "faceid_big_2.h5"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(dir_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to hdf5
    model.save_weights(dir_name_weight)

    print("saved model to disk")

def PCA_image(X_embedded, name):
    color = 0
    j = 0
    for i in range(len((X_embedded))):
        el = X_embedded[i]
        if i % 12 == 0 and not i==0:
            color+=1
            color=color%10
            j += 1
            plt.legend([labels[j - 1]])
        plt.scatter(el[0], el[1], color="C" + str(color))

    plt.savefig(SOURCE + "/handID/" + name + '.png')
    plt.gcf().clear()

def PCA_out(outputs):
    X_embedded = TSNE(2).fit_transform(outputs)
    X_embedded.shape
    X_PCA = PCA(3).fit_transform(outputs)
    print(X_PCA.shape)

    X_embedded = TSNE(2).fit_transform(X_PCA)
    print(X_embedded.shape)

    return X_embedded

gen = generator(20, train_source)
val_gen = val_generator(4, test_source)

# outputs = model_final.fit_generator(gen, steps_per_epoch=30, epochs=1, validation_data = val_gen, validation_steps=30)

cop = create_wrong(test_source+"/")
model_final.predict([cop[0].reshape((1,input_shapes[0],input_shapes[1],input_shapes[2])), cop[1].reshape((1,input_shapes[0],input_shapes[1],input_shapes[2]))])


im_in1 = Input(shape=(input_shapes[0],input_shapes[1],input_shapes[2]))
#im_in2 = input(shape=(input_shapes[0],input_shapes[1],input_shapes[2]))
feat_x1 = model_top(im_in1)
#feat_x2 = model_top(im_in2)
model_output = Model(inputs = im_in1, outputs = feat_x1)

model_output.summary()

adam = Adam(lr=0.001)
sgd = SGD(lr=0.001, momentum=0.9)
model_output.compile(optimizer=adam, loss=contrastive_loss)

cop = create_couple(test_source+"/")
model_output.predict(cop[0].reshape((1,input_shapes[0],input_shapes[1],input_shapes[2])))

def create_input(file):
  #  print(folder)

    mat1 = np.asarray(process_image(img.imread(file), factor = DOWNSCALING_FACTOR * 5))
    
    return mat1

def get_outputs(train_source = train_source, dim = 128):
    outputs=[]
    n=0
    for folder in glob.glob(train_source + "/*"):
        i=0
        for file in glob.glob(folder + '/*.jpg'):
            i+=1
            outputs.append(model_output.predict(create_input(file).reshape((1,input_shapes[0],input_shapes[1],input_shapes[2]))))
        #print(i)
        n+=1
    #    print("folder ", n, " of ", len(glob.glob('faceid_train/*')))
    #print(len(outputs))
    
    return np.asarray(outputs).reshape((-1,128))

outputs= get_outputs(train_source = visual_source)
outputs = outputs.reshape((-1,128))

X_embedded = PCA_out(outputs)

PCA_image(X_embedded, "before")

#file1 = ('faceid_train/(2012-05-16)(154211)/015_1_d.dat')
#inp1 = create_input(file1)
#file1 = ('faceid_train/(2012-05-16)(154211)/011_1_d.dat')
#inp2 = create_input(file1)

#model_final.predict([inp1, inp2])

outputs = model_final.fit_generator(generator(20, test_source), steps_per_epoch=30, epochs=int(EPOCHS * 0.5), validation_data = generator(4, test_source), validation_steps=30)
save_model_keras(model_final)

cop = create_wrong(test_source+"/")
model_final.predict([cop[0].reshape((1,input_shapes[0],input_shapes[1],input_shapes[2])), cop[1].reshape((1,input_shapes[0],input_shapes[1],input_shapes[2]))])

im_in1 = Input(shape=(input_shapes[0],input_shapes[1],input_shapes[2]))
#im_in2 = input(shape=(input_shapes[0],input_shapes[1],input_shapes[2]))
feat_x1 = model_top(im_in1)
#feat_x2 = model_top(im_in2)
model_output = Model(inputs = im_in1, outputs = feat_x1)

model_output.summary()

adam = Adam(lr=0.001)
sgd = SGD(lr=0.001, momentum=0.9)
model_output.compile(optimizer=adam, loss=contrastive_loss)

cop = create_couple(test_source+"/")
model_output.predict(cop[0].reshape((1,input_shapes[0],input_shapes[1],input_shapes[2])))

outputs= get_outputs(train_source = visual_source)
outputs = outputs.reshape((-1,128))

X_embedded = PCA_out(outputs)

PCA_image(X_embedded, "after")