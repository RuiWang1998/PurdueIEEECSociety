from Model import handCNN, handCNNDense
from testTrain import loadAndTrain, firstTrain, loadAndTest
from constants import DOWNSCALING_FACTOR, EPOCHS, learning_rate, NUM_CLASS, GROWTH_RATE, device

# device = 'cpu'
print(device)

### introducing the model
net = handCNNDense(num_class = NUM_CLASS, factor = DOWNSCALING_FACTOR, k = GROWTH_RATE).to(device)
# net = handCNN(num_class=100).to(device)

if __name__ == '__main__':
    model_name = 'modelDense3Block'
    model_dir = './models/'
    firstTrain(net, model_dir, model_name, epochs = EPOCHS)
    # loadAndTrain(model = model_name, epoch = 3, index = 4, dir = model_dir, best_accuracy = 50)
    # loadAndTest(model_dir, model_name)
    # export(model_name, model_dir, optimizer_2 = optimizer)