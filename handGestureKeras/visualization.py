import sklearn
from sklearn.manifold import TSNE

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob
import numpy as np
import matplotlib.image as img

from constants import SOURCE, DATA_SOURCE, TRAIN_FOLDER, VIS_FOLDER, resolution, DOWNSCALING_FACTOR
from imageProcessing import process_image
from dataloader import input_shape

visual_source = SOURCE+DATA_SOURCE+VIS_FOLDER
input_shapes = input_shape()

def create_input(file, factor = DOWNSCALING_FACTOR * 5):
  #  print(folder)

    mat1 = np.asarray(process_image(img.imread(file), factor=factor))
    
    return mat1

def get_outputs(model, train_source = visual_source, dim = 128, factor = DOWNSCALING_FACTOR):
    outputs=[]
    average = []
    for folder in glob.glob(train_source + "/*"):
        avg = []
        for file in glob.glob(folder + '/*.jpg'):
            output = model.predict(create_input(file, factor=factor).reshape((1,input_shapes[0],input_shapes[1],input_shapes[2])))
            outputs.append(np.asarray(output))
            avg.append(output)
        mean = np.mean(np.asarray(avg), axis=0)
        average.append(mean)
        #print(i)
    #    print("folder ", n, " of ", len(glob.glob('faceid_train/*')))
    #print(len(outputs))
    
    return np.asarray(outputs).reshape((-1,dim)), np.asarray(average)

def PCA_image(X_embedded, name, item = 196):
    color = 0
    j = 0
    for i in range(len((X_embedded))):
        el = X_embedded[i]
        if i % item == 0 and not i==0:
            color+=1
            color=color%10
            j += 1
        plt.scatter(el[0], el[1], color="C" + str(color))

    plt.savefig(SOURCE + "/handID/" + name + '.png')
    plt.gcf().clear()

def PCA_out(outputs):
    X_embedded = TSNE(2).fit_transform(outputs)
    X_PCA = PCA(3).fit_transform(outputs)

    X_embedded = TSNE(2).fit_transform(X_PCA)

    return X_embedded

def save_mean(mean, model_name):
    '''
    This function saves the mean into a csv file
    '''

    a = np.asarray(mean)
    np.savetxt("./means/" + str(model_name) + ".csv", a, delimiter=",")

    return

def load_mean(model_name):
    '''
    This function loads the mean from a csv file
    '''

    mean = np.loadtxt(open("./means/" + str(model_name) + ".csv", "r"), delimiter=",", skiprows=0)

    return mean



#def model_distance(model, pic1, pic2):
#    file1 = (pic1)
#    inp1 = create_input_rgbd(file1)
#    file1 = (pic2)
#    inp2 = create_input_rgbd(file1)

#    return model.predict([inp1, inp2])