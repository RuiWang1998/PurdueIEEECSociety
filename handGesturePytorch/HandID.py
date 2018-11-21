import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.onnx
import time

from Model import handCNN, handCNNDense, ContrastiveLoss
from couples import create_couple_batch
from constants import DOWNSCALING_FACTOR, EPOCHS, learning_rate, NUM_CLASS, GROWTH_RATE, device, SOURCE, DATA_SOURCE, TEST_FOLDER, TRAIN_FOLDER, test_count, train_count, BATCH_SIZE, TRAIN_AUG, VIS_FOLDER
from HandIDFunctions import firstTrain, loadAndTrain, loadAndTest
from visual import PCA_image, PCA_out, get_outputs, save_mean, load_mean

### introducing the model
model = handCNN(num_class = 100)
## model = handCNNDense(num_class = NUM_CLASS * 20, factor = DOWNSCALING_FACTOR, k = GROWTH_RATE)
#net = model

#print(model)

PCA_image(PCA_out(get_outputs(model, train_source=SOURCE+DATA_SOURCE+VIS_FOLDER)), name = 'before')

#model.to(device)
model_dir = './models/'
# model_name = 'handID'

model_name = 'modelDense3Block'
model, lowest_loss = firstTrain(model, model_dir, model_name, epochs = 20, train_source = SOURCE+DATA_SOURCE+VIS_FOLDER)
lowest_loss = np.Inf
# model, lowest_loss = loadAndTrain(model = model_name, epoch = 0, index = 4, dir = model_dir, lowest_loss=lowest_loss)
model = torch.load(model_dir + model_name)
outputs, mean = get_outputs(model, train_source=SOURCE+DATA_SOURCE+VIS_FOLDER)
save_mean(mean, model_name = model_name, dim = 5)
a = load_mean(model_name)

PCA_image(PCA_out(outputs), name = 'after')
# loadAndTest(model_dir, model_name)
# export(model_name, model_dir, optimizer_2 = optimizer)