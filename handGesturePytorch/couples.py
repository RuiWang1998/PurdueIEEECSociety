import numpy as np
import torch
import glob
import random
from imageProcessing import process_image
import matplotlib.pyplot as plt
import matplotlib.image as img
from constants import DOWNSCALING_FACTOR, resolution

def create_couple(file_path, folder, photo_file):
  #  print(folder)
    # print("Correct foler" + folder)

    mat1 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    #plt.imshow(mat1)
    #plt.show()

    photo_file2 = np.random.choice(glob.glob(folder + "/*.jpg"))
    while photo_file2 == photo_file:
        photo_file2 = np.random.choice(glob.glob(folder + "/*.jpg"))
    mat2 = np.asarray(process_image(img.imread(photo_file2), factor = DOWNSCALING_FACTOR * 5))
    #plt.imshow(mat2)
    #plt.show()
    return np.array([mat1, mat2]), photo_file

def create_wrong(file_path, folder, photo_file, photo = False):

    mat1 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    #plt.imshow(mat1)
    #plt.show()
    
    folder2=np.random.choice(glob.glob(file_path + "/*"))
    while folder==folder2 or folder2=="datalab": #it activates if it chose the same folder
        folder2=np.random.choice(glob.glob(file_path + "/*"))
    #print("folder2: "+folder2)
    photo_file = np.random.choice(glob.glob(folder2 + "/*.jpg"))
    mat2 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
    #plt.imshow(mat2)
    #plt.show()
  
    return np.array([mat1, mat2]), folder, photo_file

def create_couple_batch(batch_size, file_path, folders, photo_files, prob = 0.5):
    couple = []
    label = []

    for _ in range(batch_size):
        file_idx = np.random.choice(5)
        folder = folders[file_idx]
        photo_file = photo_files[file_idx]
        if random.uniform(0,1) > 0.8:
            matrix, photo_file = create_couple(file_path, folder=folder, photo_file=photo_file)
            couple.append(matrix)
            label.append(0)
        else:
            matrix, folder, photo_file = create_wrong(file_path, folder=folder, photo_file=photo_file)
            couple.append(matrix)
            label.append(1)

    return torch.tensor(np.asarray(couple)), torch.tensor(label)

def save_mean(mean, model_name):
    '''
    This function saves the mean into a csv file
    '''

    a = np.asarray(mean)
    np.savetxt("./means/" + str(model_name) + ".csv", a, delimiter=",")

    return
