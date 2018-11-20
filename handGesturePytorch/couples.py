import numpy as np
import torch
import glob
import random
from imageProcessing import process_image
import matplotlib.pyplot as plt
import matplotlib.image as img
from constants import DOWNSCALING_FACTOR, resolution

def create_couple(file_path):
    folder=np.random.choice(glob.glob(file_path + "/*"))
    while folder == "datalab":
      folder=np.random.choice(glob.glob(file_path + "/*"))
  #  print(folder)
    photo_file = np.random.choice(glob.glob(folder + "/*.jpg"))
    mat1 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    #plt.imshow(mat1)
    #plt.show()

    photo_file2 = np.random.choice(glob.glob(folder + "/*.jpg"))
    while photo_file2 == photo_file:
        photo_file2 = np.random.choice(glob.glob(folder + "/*.jpg"))
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

def create_couple_batch(batch_size, file_path, prob = 0.5):
    couple = []
    label = []
    for _ in range(batch_size):
        if random.uniform(0,1) > 0.5:
            couple.append(create_couple(file_path))
            label.append(0)
        else:
            couple.append(create_wrong(file_path))
            label.append(1)

    return torch.tensor(np.asarray(couple)), torch.tensor(label)
