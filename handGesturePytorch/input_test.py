import numpy as np
import csv
import torch
from visual import save_mean, save_mean_2d, euclidean_distance, euclidean_data, create_input
from constants import acceptable_range, SOURCE, DATA_SOURCE, TRAIN_FOLDER, VIS_FOLDER, resolution, DOWNSCALING_FACTOR, VIS_SMALLER, TEST_FOLDER
input_shapes = (3, int(resolution[0] * DOWNSCALING_FACTOR), int(resolution[1] * DOWNSCALING_FACTOR))

import glob

# load means
means = csv.reader(open("./means/handIDmean.csv", "r"), delimiter=",")
means = list(means)
means = np.array(means).astype("float")


model_name = 'handID'
model_dir = './models/'
model = torch.load(model_dir + model_name).to('cpu')

#def train_a_little

outputs = []
def test_final(model, folder):
    '''
    This function creates new labels in the model

    model:      the model that does the prediction
    folder:     the folder in which the new pictures are stored
    '''

    result = []
    output_average = []
    with torch.no_grad():
        for file in glob.glob(folder + '/5' + '/*.jpg'):
            inputs = torch.tensor(create_input(file, factor = DOWNSCALING_FACTOR), dtype=torch.float).transpose(0, 2).transpose(1, 2).reshape(1, input_shapes[0], input_shapes[1], input_shapes[2])
            output = model(inputs).numpy()
            eu_distances = []
            output_average.append(output)
            for mean in means:
                displacement = output - mean
                eu_dis = euclidean_distance(displacement)
                eu_distances.append(eu_dis)
            result.append(np.argmin(np.asarray(eu_distances)))

        new_mean = np.mean(np.asarray(output_average), axis=0)

    return result, new_mean

result, new_mean = test_final(model, SOURCE+DATA_SOURCE+VIS_FOLDER)

if __name__ == '__main__':
    means = np.append(means, new_mean, axis=0)
    save_mean_2d(means, model_name + 'mean')
    save_mean_2d(result, 'new')