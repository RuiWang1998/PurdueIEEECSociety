import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.onnx
import time
import glob

from Model import handCNN, handCNNDense, ContrastiveLoss
from couples import create_couple_batch, save_mean
from constants import DOWNSCALING_FACTOR, EPOCHS, learning_rate, NUM_CLASS, GROWTH_RATE, device, SOURCE, DATA_SOURCE, TEST_FOLDER, TRAIN_FOLDER, test_count, train_count, BATCH_SIZE, ALL_FOLDER
from HandIDFunctions import firstTrain, loadAndTrain, loadAndTest
from visual import PCA_image, PCA_out, get_outputs


### introducing the model
model = handCNN(num_class = 100)
# model = handCNNDense(num_class = NUM_CLASS * 20, factor = DOWNSCALING_FACTOR, k = GROWTH_RATE)
# net = model

#print(model)

# outputs, mean = get_outputs(net)
# PCA_image(PCA_out(outputs), name = 'before')


folders = glob.glob(SOURCE+DATA_SOURCE+TRAIN_FOLDER + "/*")
photo_files=[]
for folder in folders:
    photo_files.append(np.random.choice(glob.glob(folder + "/*.jpg")))
    print(photo_files)

# model.to(device)
model_name = 'handID'
model_dir = './models/'
# model = torch.load(model_dir + model_name).to('cpu')
# model, lowest_loss = firstTrain(model, model_dir, model_name, folders=folders, photo_files=photo_files, epochs = 400)
lowest_loss = np.Inf
model, lowest_loss = loadAndTrain(model = model_name, epochs = 1000, index = 4, dir = model_dir, lowest_loss=lowest_loss, folders=folders, photo_files=photo_files)
model.to('cpu')
outputs, mean, max = get_outputs(model, train_source=SOURCE+DATA_SOURCE+TRAIN_FOLDER, factor=DOWNSCALING_FACTOR * 5)
save_mean(outputs, model_name + 'all')
save_mean(mean[:, 0, :], model_name + 'mean')
save_mean(max[:, 0, :], model_name + 'max')
PCA_image(PCA_out(outputs), name = 'after')
# loadAndTest(model_dir, model_name)
# export(model_name, model_dir, optimizer_2 = optimizer)