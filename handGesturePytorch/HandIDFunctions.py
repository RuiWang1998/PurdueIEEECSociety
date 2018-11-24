import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.onnx
import time
import glob

from Model import handCNN, handCNNDense, ContrastiveLoss
from couples import create_couple_batch
from constants import DOWNSCALING_FACTOR, EPOCHS, learning_rate, NUM_CLASS, GROWTH_RATE, device, SOURCE, DATA_SOURCE, TEST_FOLDER, TRAIN_FOLDER, test_count, train_count, BATCH_SIZE, optimizer


def train(model, folders, photo_files, device = device, batch_size = 15, loss_func = ContrastiveLoss(), epoch = EPOCHS, train_source = SOURCE+DATA_SOURCE+TRAIN_FOLDER, epochs = EPOCHS):
    '''
    This function performs the training step for just one epoch
    '''
    model.train()
    optim = optimizer(model)
    steps = int(3 * train_count/batch_size)

    for batch_idx in range(steps):
        data, target = create_couple_batch(batch_size, train_source, folders, photo_files)
        data, target = data.to(device = device, dtype=torch.float), target.to(device)
        optim.zero_grad()
        output1 = model(data[:, 0, :, :, :].transpose(1, 3).transpose(2, 3))
        output2 = model(data[:, 1, :, :, :].transpose(1, 3).transpose(2, 3))
        loss = loss_func(output1, output2, target)
        loss.backward()
        optim.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, epochs, batch_idx, steps,
                100. * batch_idx / steps, loss.item()))

    return

def test(model, folders, photo_files, device = device, loss_func = ContrastiveLoss(), save = False, train_source = SOURCE+DATA_SOURCE+TRAIN_FOLDER, test_source = SOURCE+DATA_SOURCE+TEST_FOLDER):
    '''
    This function tests the model's capacity
    '''

    model.eval()
    test_loss = 0
    train_loss = 0
    with torch.no_grad():
        data, target = create_couple_batch(80, train_source, folders, photo_files)
        data, target = data.to(device = device, dtype=torch.float), target.to(device)
        output1 = model(data[:, 0, :, :, :].transpose(1, 3).transpose(2, 3))
        output2 = model(data[:, 1, :, :, :].transpose(1, 3).transpose(2, 3))
        test_loss += loss_func(output1, output2, target)
        test_loss, output1, output2, data, target = test_loss.to('cpu'), output1.to('cpu'), output2.to(device='cpu'), data.to(device = 'cpu', dtype=torch.float), target.to(device='cpu')

        data, target = create_couple_batch(200, train_source, folders, photo_files)
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
    
def firstTrain(net, output_dir, output_file, folders, photo_files, epochs = EPOCHS, train_source = SOURCE+DATA_SOURCE+TRAIN_FOLDER):
    '''
    This function is called to train a model uninitialized
    '''
    print(net)

    net.to(device)
    start = time.time()
    # Train the model
    lowest_loss = np.Inf

    start = time.time()
    total_step = int(train_count/BATCH_SIZE)

    # prepare the array from which the data is about to be plotted
    t_plot = np.zeros(epochs)
    train_loss_plot = np.zeros(epochs)
    test_loss_plot = np.zeros(epochs)



    # trainingy
    print("start training")
    time_step = time.time()
    for epoch in range(epochs):

        t_plot[epoch] = epoch
        train(net, folders=folders, photo_files=photo_files, device=device, batch_size=BATCH_SIZE, epoch=epoch, train_source = train_source, epochs=epochs)
        train_loss_plot[epoch], test_loss_plot[epoch] = test(net, folders=folders, photo_files=photo_files)
        print("Epoch: {}| time elapsed: {} s\n".format(epoch, time.time()-time_step))
        time_step = time.time()

        if test_loss_plot[epoch] <= lowest_loss:
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

def loadAndTrain(model, dir, folders, photo_files, epochs = EPOCHS, index = 1, lowest_loss = np.Inf, train_source = SOURCE+DATA_SOURCE+TRAIN_FOLDER):
    '''
    This function is called when training a model loaded from a file
    '''
    net = torch.load(dir + model)
    print(net)
    optimizer_2 = optimizer(net)
    time_step = time.time()

    for epoch in range(epochs):

        train(net, folders=folders, photo_files=photo_files, epoch = epoch, batch_size = BATCH_SIZE, epochs = epochs)
        _, test_loss = test(net, folders=folders, photo_files=photo_files)
        if test_loss < lowest_loss:
            lowest_loss = test_loss
            torch.save(net, dir + model)
        print("Epoch: {}| time elapsed: {} s\n".format(epoch, time.time()-time_step))
        time_step = time.time()

    return net, lowest_loss

def loadAndTest(dir, model):
    '''
    This function loads a model and test its performance
    '''

    net = torch.load(dir + model)
    print(net)
    test(net)

#def test_all(model, source):
#    data, target = create_couple_batch(1, source)
#    data, target = data.to(device = 'cpu', dtype=torch.float), target.to('cpu')
#    output1 = model(data[:, 0, :, :, :].transpose(1, 3).transpose(2, 3))
#    output2 = model(data[:, 1, :, :, :].transpose(1, 3).transpose(2, 3))
#    target
#    pdist(output1, output2)

#    for folder in glob.glob(train_source + "/*"):
#        print("Loading from folder" + folder)
#        data = []
#        for file in glob.glob(folder + '/*.jpg'):
#            mat1 = np.asarray(process_image(img.imread(photo_file), factor = DOWNSCALING_FACTOR * 5))
#            data.append()