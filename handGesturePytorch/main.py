import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import glob
import csv
import math
import platform

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from skimage import data, color
from Model import handCNNDense, handCNN
from dataloader import generic_transform, new_transform
import random
from constants import DOWNSCALING_FACTOR, TRAIN_FOLDER, TEST_FOLDER, ALL_FOLDER, PARENT_FOLDER_NAME, SOURCE, EPOCHS, BATCH_SIZE, learning_rate, NUM_CLASS, DATA_SOURCE, TEST_AUG, TRAIN_AUG


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

### this section prepross the data

print("loading the data")

train_dataset = torchvision.datasets.ImageFolder(root=(SOURCE + DATA_SOURCE + TRAIN_AUG +'/'), 
                                                     transform=transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(root=(SOURCE + DATA_SOURCE + TEST_FOLDER +'/'), 
                                                    transform=transforms.ToTensor())
all_dataset = torchvision.datasets.ImageFolder(root=(SOURCE + DATA_SOURCE + ALL_FOLDER +'/'), 
                                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False)
all_loader = torch.utils.data.DataLoader(dataset=all_dataset,
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False)

print("data loading finished, starting the model")
### introducing the model
net = handCNNDense(num_class = NUM_CLASS, factor = DOWNSCALING_FACTOR).to(device)
# net = handDenseNet().to(device)
loss_func = nn.CrossEntropyLoss()       # we are only using cross entropy loss for now, we would love to add Wasserstein distance into the loss function later on to smooth the update
optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.99, 0.999), 
                           eps=1e-08, weight_decay=0, amsgrad=True)


def train(model, device, train_loader, optimizer, loss_func, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 33 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return

def test(model, device, test_loader, loss_func, train_loader, best_test_accuracy):
    model.eval()
    test_loss = 0
    train_loss = 0
    correct_test = 0    
    correct_train = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct_test += pred.eq(target.view_as(pred)).sum().item()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += loss_func(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct_train += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('\nTest set:     Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct_test, len(test_loader.dataset),
        100. * correct_test / len(test_loader.dataset)))
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct_train, len(train_loader.dataset),
        100. * correct_train / len(train_loader.dataset)))

    return 100. * correct_train / len(train_loader.dataset), 100. * correct_test / len(test_loader.dataset)
    
def firstTrain(epochs = EPOCHS):
    start = time.time()
    # Train the model
    best_accuracy = 0

    start = time.time()
    total_step = len(train_loader)

    # prepare the array from which the data is about to be plotted
    t_plot = np.zeros(epochs + 1)
    train_accuracy_plot = np.zeros(epochs + 1)
    test_accuracy_plot = np.zeros(epochs + 1)

    # training
    train_accuracy_plot[0], test_accuracy_plot[0] = test(net, device, test_loader, loss_func, train_loader, best_accuracy)
    print("start training")
    for epoch in range(epochs):
        t_plot[epoch + 1] = epoch
        train(net, device, train_loader, optimizer, loss_func, epoch)
        train_accuracy_plot[epoch + 1], test_accuracy_plot[epoch + 1] = test(net, device, test_loader, loss_func, train_loader, best_accuracy)

        if test_accuracy_plot[epoch] >= best_accuracy:
            best_accuracy = test_accuracy_plot[epoch]
            best_correct = round(best_accuracy * len(test_loader.dataset))
            test_volume = len(test_loader.dataset)
            dir = './models/'
            filename = dir + 'modelDenseAug'
            torch.save(net, filename)
    
    plt.plot(t_plot, train_accuracy_plot, 'b', t_plot, test_accuracy_plot, 'r')
    plt.title("Accuracy over epochs")
    plt.ylabel('The accuracy of test and training sets over epochs')
    plt.xlabel('Epochs')
    plt.legend(['training set','test set'])
    plt.savefig(IMAGE_DIR + str(int(best_correct/100)) + '_' + str(test_volume) + '_' + str(round(best_accuracy)) + '_' + '.png')

def loadAndTrain(model, dir, epoch = EPOCHS, index = 1, optimizer_2 = optimizer, best_accuracy = 90):
    net = torch.load(dir + model)
    for epochs in range(epoch):
        train(net, device, train_loader, optimizer_2, loss_func, epochs)
        _, test_accuracy = test(net, device, test_loader, loss_func, train_loader, best_accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(net, dir + model)
            
    return

firstTrain(epochs = EPOCHS)
# loadAndTrain(model = 'modelDenseAug', epoch = 30, index = 4, optimizer_2 = optimizer, dir = './models/')