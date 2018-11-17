# this code is from https://github.com/normandipalo/faceID_beta/blob/master/faceid_beta.ipynb with a few modifications
import numpy as np
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
    mat1=(mat1-np.mean(mat1))/np.max(mat1)
    plt.imshow(mat1)
    plt.show()

    photo_file = np.random.choice(glob.glob(folder + "/*.jpg"))
    mat2 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    mat2=(mat2-np.mean(mat2))/np.max(mat2)
    plt.imshow(mat2)
    plt.show()
    return np.array([mat1, mat2])

def create_wrong(file_path):
    folder=np.random.choice(glob.glob(file_path + "/*"))
    while folder == "datalab":
      folder=np.random.choice(glob.glob(file_path + "/*"))    
    photo_file = np.random.choice(glob.glob(folder + "/*.jpg"))
    mat1 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    mat1=(mat1-np.mean(mat1))/np.max(mat1)
    plt.imshow(mat1)
    plt.show()
    
    folder2=np.random.choice(glob.glob(file_path + "/*"))
    while folder==folder2 or folder2=="datalab": #it activates if it chose the same folder
        folder2=np.random.choice(glob.glob(file_path + "/*"))
    photo_file = np.random.choice(glob.glob(folder2 + "/*.jpg"))
    mat2 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    mat2=(mat2-np.mean(mat2))/np.max(mat2)
    plt.imshow(mat2)
    plt.show()
  
    return np.array([mat1, mat2])

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sum((K.square(u - v)), axis=1, keepdims=True)

def contrastive_loss(y_true,y_pred):
    margin=1.
    # return K.mean( K.square(y_pred) )
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))