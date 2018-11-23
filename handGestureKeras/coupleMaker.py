# this code is from https://github.com/normandipalo/faceID_beta/blob/master/faceid_beta.ipynb with a few modifications
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from constants import DOWNSCALING_FACTOR, resolution
from keras import backend as K
from imageProcessing import process_image

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

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sum((K.square(u - v)), axis=1, keepdims=True)

def contrastive_loss(y_true,y_pred):
    margin=1.
    # return K.mean( K.square(y_pred) )

    ## parts of the code came from https://github.com/normandipalo/faceID_beta/blob/master/faceid_beta.ipynb

    self_defined = K.sum(1. - y_true) * K.square(y_pred) + y_true * 1000./K.square(y_pred)
    original = K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))
    return self_defined

def soft_max_crossEntropy(y_true, y_pred):

    return K.categorical_crossentropy(y_true, K.softmax(y_pred))

def vanilla_loss(y_true, inputs):

    return inputs


def create_input(file):
  #  print(folder)

    mat1 = np.asarray(process_image(img.imread(file), factor = DOWNSCALING_FACTOR * 5))
    
    return mat1