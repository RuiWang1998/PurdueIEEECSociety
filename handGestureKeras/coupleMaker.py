# this code is from https://github.com/normandipalo/faceID_beta/blob/master/faceid_beta.ipynb with a few modifications
import numpy as np
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
from constants import DOWNSCALING_FACTOR, resolution
from keras import backend as K

def create_couple(file_path):
    folder=np.random.choice(glob.glob(file_path + "*"))
    while folder == "datalab":
      folder=np.random.choice(glob.glob(file_path + "*"))
  #  print(folder)
    mat=np.zeros((resolution[0] * DOWNSCALING_FACTOR, resolution[1] * DOWNSCALING_FACTOR), dtype='float32')
    i=0
    j=0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue 
                if int(val) > 1200 or int(val) == -1: val= 1200
                mat[i][j]=float(int(val))
                j+=1
                j=j%resolution[1] * DOWNSCALING_FACTOR

            i+=1
        mat = np.asarray(mat)
    mat=(mat-np.mean(mat))/np.max(mat)
#    plt.imshow(mat)
#    plt.show()
    
    mat2=np.zeros((resolution[0] * DOWNSCALING_FACTOR, resolution[1] * DOWNSCALING_FACTOR), dtype='float32')
    i=0
    j=0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue 
                if int(val) > 1200 or int(val) == -1: val= 1200
                mat2[i][j]=float(int(val))
                j+=1
                j=j%resolution[1] * DOWNSCALING_FACTOR

            i+=1
        mat2 = np.asarray(mat2)
    mat2=(mat2-np.mean(mat2))/np.max(mat2)
#    plt.imshow(mat2)
#    plt.show()
    return np.array([mat, mat2])

def create_wrong(file_path):
    folder=np.random.choice(glob.glob(file_path + "*"))
    while folder == "datalab":
      folder=np.random.choice(glob.glob(file_path + "*"))    
    mat=np.zeros((resolution[0] * DOWNSCALING_FACTOR, resolution[1] * DOWNSCALING_FACTOR), dtype='float32')
    i=0
    j=0
    depth_file = np.random.choice(glob.glob(folder + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue 
                if int(val) > 1200 or int(val) == -1: val= 1200
                mat[i][j]=float(int(val))
                j+=1
                j=j%resolution[1] * DOWNSCALING_FACTOR

            i+=1
        mat = np.asarray(mat)
    mat=(mat-np.mean(mat))/np.max(mat)
 #   plt.imshow(mat)
 #   plt.show()
    

    folder2=np.random.choice(glob.glob(file_path + "*"))
    while folder==folder2 or folder2=="datalab": #it activates if it chose the same folder
        folder2=np.random.choice(glob.glob(file_path + "*"))
    mat2=np.zeros((resolution[0] * DOWNSCALING_FACTOR, resolution[1] * DOWNSCALING_FACTOR), dtype='float32')
    i=0
    j=0
    depth_file = np.random.choice(glob.glob(folder2 + "/*.dat"))
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue
                if int(val) > 1200 or int(val) == -1: val= 1200
                mat2[i][j]=float(int(val))
                j+=1
                j=j%resolution[1] * DOWNSCALING_FACTOR

            i+=1
        mat2 = np.asarray(mat2)
    mat2=(mat2-np.mean(mat2))/np.max(mat2)
 #   plt.imshow(mat2)
 #   plt.show()
  
    
    return np.array([mat, mat2])

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))