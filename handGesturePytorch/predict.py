import torch
from visual import save_mean, save_mean_2d, euclidean_distance, euclidean_data, create_input
import numpy as np
import csv
import torch
from constants import acceptable_range, SOURCE, DATA_SOURCE, TRAIN_FOLDER, VIS_FOLDER, resolution, DOWNSCALING_FACTOR, VIS_SMALLER, TEST_FOLDER, acceptable_range

input_shapes = (3, int(resolution[0] * DOWNSCALING_FACTOR), int(resolution[1] * DOWNSCALING_FACTOR))

import glob



def predict(model, inputs):

    with torch.no_grad():
        inputs = torch.tensor(inputs)
        output = model(inputs).numpy()
        eu_distances = []
        for mean in means:
            displacement = output - mean
            eu_dis = euclidean_distance(displacement)
            eu_distances.append(eu_dis)

        min = np.min(np.asarray(eu_distances))
        if min < acceptable_range:
            certainty = True
            min_idx = np.argmin(np.asarray(eu_distances))
        else:
            certainty = False
            min_idx = np.argmin(np.asarray(eu_distances))
        
    return min_idx, certainty

if __name__ == '__main__':

    all_output = csv.reader(open("./means/handIDall.csv", "r"), delimiter=",")
    all_output = list(all_output)
    all_output = np.array(all_output).astype("float")

    means = csv.reader(open("./means/handIDmean.csv", "r"), delimiter=",")
    means = list(means)
    means = np.array(means).astype("float")

    model_name = 'handID'
    model_dir = './models/'
    model = torch.load(model_dir + model_name).to('cpu')

    min_idx, certainty = predict(model, torch.tensor(create_input(SOURCE+DATA_SOURCE+VIS_FOLDER+'/testcapture0.jpg', factor = DOWNSCALING_FACTOR), dtype=torch.float).transpose(0, 2).transpose(1, 2).reshape(1, input_shapes[0], input_shapes[1], input_shapes[2]))
    if certainty:
        print("The class is {} with certainty".format(min_idx))
    else:
        print("The class is {} with no certainty".format(min_idx))