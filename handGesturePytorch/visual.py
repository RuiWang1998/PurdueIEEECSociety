import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import glob
import numpy as np
import matplotlib.image as img

from constants import SOURCE, DATA_SOURCE, TRAIN_FOLDER, VIS_FOLDER, resolution, DOWNSCALING_FACTOR
from imageProcessing import process_image

visual_source = SOURCE+DATA_SOURCE+VIS_FOLDER
input_shapes = (3, int(resolution[0] * DOWNSCALING_FACTOR), int(resolution[1] * DOWNSCALING_FACTOR))

def create_input(file):
  #  print(folder)

    mat1 = np.asarray(process_image(img.imread(file), factor = DOWNSCALING_FACTOR * 5))
    
    return mat1

def get_outputs(model, train_source = visual_source, dim = 128):
    model.eval()
    with torch.no_grad():
        outputs=[]
        for folder in glob.glob(train_source + "/*"):
            i=0
            for file in glob.glob(folder + '/*.jpg'):
                i+=1
                outputs.append(model(torch.tensor(create_input(file).reshape((1,input_shapes[0],input_shapes[1],input_shapes[2])), dtype=torch.float)).numpy())
                #print(i)
        #    print("folder ", n, " of ", len(glob.glob('faceid_train/*')))
        #print(len(outputs))
        outputs = np.asarray(outputs)
    return outputs.reshape((-1,100))

def PCA_image(X_embedded, name):
    color = 0
    j = 0
    for i in range(len((X_embedded))):
        el = X_embedded[i]
        if i % 12 == 0 and not i==0:
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