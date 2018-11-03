import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision.models.densenet import DenseNet
from torchvision import datasets, models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import glob
import csv
import math

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from skimage import data, color
from imageProcessing import load_images, show_image, preprocessing, normalize_images
from Model import handCNN
from dataloader import generic_transform, new_transform
import random


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

### this is for the initialization of some constant and seed
# this section defines constants
DOWSCALING_FACTOR = 0.2                           # this means the number of pixels are reduced to downscaling_factor ^ 2 time of its orginal value
TRAIN_FOLDER = 'image_train_folder'               # this is where the processed image goes
TEST_FOLDER = 'image_test_folder'
PARENT_FOLDER_NAME = 'image_folder'               # this is the parent folder 
SOURCE_WINDOWS = 'C:/'
SOURCE_LINUX = '/mnt/c/'
SECOND_SOURCE = 'Users/Rui/source/repos/handGestureRecogIEEE/handGestureRecogIEEE/'
torch.manual_seed(1)                              # this controls the random seed so that the result is reproducible
random.seed(1)
NUM_CLASS = 5
TEST_PORTION = 0.8

# this needs to change if the platform is changed
SOURCE = SOURCE_WINDOWS + SECOND_SOURCE

# Hyper parameters
EPOCHS = 30
BATCH_SIZE = 3
learning_rate = 0.0001

### this section prepross the data
test_size = 218
# test_size = preprocessing(PARENT_FOLDER_NAME, SOURCE, TRAIN_FOLDER, TEST_FOLDER, DOWSCALING_FACTOR, prob = TEST_PORTION)

train_dataset = torchvision.datasets.ImageFolder(root=(SOURCE + TRAIN_FOLDER +'/'), 
                                                     transform=transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(root=(SOURCE + TEST_FOLDER +'/'), 
                                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False)

### introducing the model
net = handCNN(num_class = NUM_CLASS, factor = DOWSCALING_FACTOR).to(device)
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
    t_plot = np.zeros(epochs)
    train_accuracy_plot = np.zeros(epochs + 1)
    test_accuracy_plot = np.zeros(epochs + 1)

    # training
    train_accuracy_plot[0], test_accuracy_plot[0] = test(net, device, test_loader, loss_func, train_loader, best_accuracy)
    for epoch in range(epochs):
        t_plot[epoch + 1] = epoch
        train(net, device, train_loader, optimizer, loss_func, epoch)
        train_accuracy_plot[epoch + 1], test_accuracy_plot[epoch + 1] = test(net, device, test_loader, loss_func, train_loader, best_accuracy)

        if test_accuracy_plot[epoch] >= best_accuracy:
            best_accuracy = test_accuracy_plot[epoch]
            dir = './models/'
            filename = dir + 'model' + str(best_accuracy)
            torch.save(net, filename)
    
    plt.plot(t_plot, train_accuracy_plot, 'b', t_plot, test_accuracy_plot, 'r')
    plt.title("Accuracy over epochs")
    plt.ylabel('The accuracy of test and training sets over epochs')
    plt.xlabel('Epochs')
    plt.legend(['training set','test set'])
    plt.savefig('test_train' + str(int(best_accuracy)) + '.png')
    #myFile = open('loss_curve.csv', 'w')
    #with myFile:  
    #   writer = csv.writer(myFile)
    #   data = [x_plot, loss_plot]
    #   writer.writerows(data)

def loadAndTrain(epoch = EPOCHS, index = 1, optimizer_2 = optimizer, best_accuracy = 90):
    net = torch.load('model'+str(index-1))
    for epochs in range(epoch):
        train(net, device, train_loader, optimizer, loss_func, epochs)
        test_accuracy = test(net, device, test_loader, loss_func, train_loader, best_accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(net, './model1')

if __name__ == '__main__':
	firstTrain(epochs = 15)
    #loadAndTrain(epoch = 30, index = 4, optimizer_2 = optimizer)