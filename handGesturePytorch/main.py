from Model import handCNNDense, handCNN
from testTrain import loadAndTrain, firstTrain, loadAndTest
from constants import DOWNSCALING_FACTOR, EPOCHS, learning_rate, NUM_CLASS, GROWTH_RATE, device

# device = 'cpu'
print(device)

### introducing the model
# net = handCNN().to(device)
net = handCNNDense(num_class = NUM_CLASS, factor = DOWNSCALING_FACTOR, k = GROWTH_RATE).to(device)

if __name__ == '__main__':
    model_name = 'modelDenseWithAug'
    model_dir = './models/'
    # firstTrain(net, model_dir, model_name, epochs = EPOCHS)
    loadAndTrain(model = model_name, epoch = 2, index = 4, dir = model_dir, best_accuracy = 50)
    # loadAndTest(model_dir, model_name)
    # export(model_name, model_dir, optimizer_2 = optimizer)