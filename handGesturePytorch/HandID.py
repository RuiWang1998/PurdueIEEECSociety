import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.onnx
import time

from Model import handCNN, handCNNDense, ContrastiveLoss
from couples import create_couple_batch
from constants import DOWNSCALING_FACTOR, EPOCHS, learning_rate, NUM_CLASS, GROWTH_RATE, device, SOURCE, DATA_SOURCE, TEST_FOLDER, TRAIN_FOLDER, test_count, train_count, BATCH_SIZE
from HandIDFunctions import firstTrain, loadAndTrain, loadAndTest
from visual import PCA_image, PCA_out, get_outputs


### introducing the model
model = handCNN(num_class = 100)
# model = handCNNDense(num_class = NUM_CLASS * 20, factor = DOWNSCALING_FACTOR, k = GROWTH_RATE)
net = model

print(model)

PCA_image(PCA_out(get_outputs(net)), name = 'before')

model.to(device)
model_name = 'handID'
model_dir = './models/'
# model, lowest_loss = firstTrain(model, model_dir, model_name, epochs = 20)
lowest_loss = np.Inf
model, lowest_loss = loadAndTrain(model = model_name, epoch = 400, index = 4, dir = model_dir, lowest_loss=lowest_loss)
model.to('cpu')
PCA_image(PCA_out(get_outputs(model)), name = 'after')
# loadAndTest(model_dir, model_name)
# export(model_name, model_dir, optimizer_2 = optimizer)