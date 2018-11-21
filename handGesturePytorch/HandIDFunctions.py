import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.onnx
import time

from Model import handCNN, handCNNDense, ContrastiveLoss
from couples import create_couple_batch
from constants import DOWNSCALING_FACTOR, EPOCHS, learning_rate, NUM_CLASS, GROWTH_RATE, device, SOURCE, DATA_SOURCE, TEST_FOLDER, TRAIN_FOLDER, test_count, train_count, BATCH_SIZE, optimizer


def train(model, device = device, batch_size = 15, loss_func = ContrastiveLoss(), epoch = EPOCHS, train_source = SOURCE+DATA_SOURCE+TRAIN_FOLDER):
    model.train()
    optim = optimizer(model)
    steps = int(3 * train_count/batch_size)
    for batch_idx in range(steps):
        data, target = create_couple_batch(batch_size, train_source)
        data, target = data.to(device = device, dtype=torch.float), target.to(device)
        optim.zero_grad()
        output1 = model(data[:, 0, :, :, :].transpose(1, 3).transpose(2, 3))
        output2 = model(data[:, 1, :, :, :].transpose(1, 3).transpose(2, 3))
        loss = loss_func(output1, output2, target)
        loss.backward()
        optim.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx, steps,
                100. * batch_idx / steps, loss.item()))

    return

def test(model, device = device, loss_func = ContrastiveLoss(), save = False, train_source = SOURCE+DATA_SOURCE+TRAIN_FOLDER, test_source = SOURCE+DATA_SOURCE+TEST_FOLDER):
    model.eval()
    test_loss = 0
    train_loss = 0
    with torch.no_grad():
        data, target = create_couple_batch(15, train_source)
        data, target = data.to(device = device, dtype=torch.float), target.to(device)
        output1 = model(data[:, 0, :, :, :].transpose(1, 3).transpose(2, 3))
        output2 = model(data[:, 1, :, :, :].transpose(1, 3).transpose(2, 3))
        test_loss += loss_func(output1, output2, target)
        test_loss, output1, output2, data, target = test_loss.to('cpu'), output1.to('cpu'), output2.to(device='cpu'), data.to(device = 'cpu', dtype=torch.float), target.to(device='cpu')

        data, target = create_couple_batch(200, test_source)
        data, target = data.to(device = device, dtype=torch.float), target.to(device)
        output1 = model(data[:, 0, :, :, :].transpose(1, 3).transpose(2, 3))
        output2 = model(data[:, 1, :, :, :].transpose(1, 3).transpose(2, 3))
        train_loss += loss_func(output1, output2, target)
        train_loss, output1, output2, data, target = train_loss.to('cpu'), output1.to('cpu'), output2.to(device='cpu'), data.to(device = 'cpu', dtype=torch.float), target.to(device='cpu')


    print('Test set:     Average loss: {:.4f}'.format(
        test_loss))
    print('Training set: Average loss: {:.4f}'.format(
        train_loss))

    return train_loss, test_loss
    
def firstTrain(net, output_dir, output_file, epochs = EPOCHS, train_source = SOURCE+DATA_SOURCE+TRAIN_FOLDER):
    print(net)
    net.to(device)
    start = time.time()
    # Train the model
    lowest_loss = np.Inf

    start = time.time()
    total_step = int(train_count/BATCH_SIZE)

    # prepare the array from which the data is about to be plotted
    t_plot = np.zeros(epochs + 1)
    train_loss_plot = np.zeros(epochs + 1)
    test_loss_plot = np.zeros(epochs + 1)

    # training
    train_loss_plot[0], test_loss_plot[0] = test(net)
    print("start training")
    time_step = time.time()
    for epoch in range(epochs):
        t_plot[epoch + 1] = epoch
        train(net, device, batch_size=BATCH_SIZE, epoch=epoch, train_source = train_source)
        train_loss_plot[epoch + 1], test_loss_plot[epoch + 1] = test(net)
        print("Epoch: {}| time elapsed: {} s\n".format(epoch, time.time()-time_step))
        time_step = time.time()

        if test_loss_plot[epoch + 1] <= lowest_loss:
            lowest_loss = test_loss_plot[epoch]
            test_volume = test_count
            dir = output_dir
            filename = dir + output_file
            torch.save(net, filename)
    
    plt.plot(t_plot, train_loss_plot, 'b', t_plot, test_loss_plot, 'r')
    plt.title("Accuracy over epochs")
    plt.ylabel('The accuracy of test and training sets over epochs')
    plt.xlabel('Epochs')
    plt.legend(['training set','test set'])
    plt.savefig(SOURCE + str(int(lowest_loss/100)) + '_' + str(test_volume) + '_' + str(round(lowest_loss)) + '_' + '.png')
    plt.gcf().clear()
    return net, lowest_loss

def loadAndTrain(model, dir, epoch = EPOCHS, index = 1, lowest_loss = np.Inf):
    net = torch.load(dir + model)
    print(net)
    optimizer_2 = optimizer(net)
    time_step = time.time()
    for epochs in range(epoch):
        train(net, epoch = epochs, batch_size = BATCH_SIZE)
        _, test_loss = test(net)
        if test_loss < lowest_loss:
            lowest_loss = test_loss
            torch.save(net, dir + model)
        print("Epoch: {}| time elapsed: {} s\n".format(epoch, time.time()-time_step))
        time_step = time.time()

    return net, lowest_loss

def loadAndTest(dir, model):
    net = torch.load(dir + model)
    print(net)
    test(net)