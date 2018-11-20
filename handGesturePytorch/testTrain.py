import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.onnx

from dataloader import train_loader, test_loader, all_loader
from constants import EPOCHS, optimizer, device, loss_func, SOURCE

def train(model, device, train_loader, loss_func = loss_func, epoch = EPOCHS):
    model.train()
    optim = optimizer(model)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optim.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optim.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return

def test(model, device, test_loader, loss_func, train_loader, save = False):
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
        if save == True:
            torch.onnx.export(model, data, "onnx.onnx")

    test_loss /= len(test_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('\nTest set:     Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct_test, len(test_loader.dataset),
        100. * correct_test / len(test_loader.dataset)))
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct_train, len(train_loader.dataset),
        100. * correct_train / len(train_loader.dataset)))

    return 100. * correct_train / len(train_loader.dataset), 100. * correct_test / len(test_loader.dataset)
    
def firstTrain(net, output_dir, output_file, epochs = EPOCHS):
    print(net)
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
    time_step = time.time()
    for epoch in range(epochs):
        t_plot[epoch + 1] = epoch
        train(net, device, train_loader, loss_func, epoch)
        train_accuracy_plot[epoch + 1], test_accuracy_plot[epoch + 1] = test(net, device, test_loader, loss_func, train_loader)
        print("Epoch: {} time elapsed {} s".format(epoch, time.time()-time_step))
        time_step = time.time()

        if test_accuracy_plot[epoch + 1] >= best_accuracy:
            best_accuracy = test_accuracy_plot[epoch]
            best_correct = round(best_accuracy * len(test_loader.dataset))
            test_volume = len(test_loader.dataset)
            dir = output_dir
            filename = dir + output_file
            torch.save(net, filename)
    
    plt.plot(t_plot, train_accuracy_plot, 'b', t_plot, test_accuracy_plot, 'r')
    plt.title("Accuracy over epochs")
    plt.ylabel('The accuracy of test and training sets over epochs')
    plt.xlabel('Epochs')
    plt.legend(['training set','test set'])
    plt.savefig(SOURCE + str(int(best_correct/100)) + '_' + str(test_volume) + '_' + str(round(best_accuracy)) + '_' + '.png')
    plt.gcf().clear()

def loadAndTrain(model, dir, epoch = EPOCHS, index = 1, best_accuracy = 90):
    net = torch.load(dir + model)
    print(net)
    optimizer_2 = optimizer(net)
    for epochs in range(epoch):
        train(net, device, train_loader, loss_func, epochs)
        _, test_accuracy = test(net, device, test_loader, loss_func, train_loader)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(net, dir + model)
            
    return

def loadAndTest(dir, model):
    net = torch.load(dir + model)
    print(net)
    test(net, device, all_loader, loss_func, all_loader)

def export(model, dir):
    net = torch.load(dir + model)
    test(net, device, test_loader, loss_func, test_loader, 90., save = True)
