import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.densenet as Dense

# this is a simple model where the number of input channels are 3, the kernel size should be 2-by-2 and the stride is 1, where the padding should only be adjusted according to the kernel size
# input of this model should be times of size 48 * 64 * 3, a colored, down sampling group
# this model contains 4 convolution-BatchNorm-ReLU-MaxPool block and one linear layer
class handCNNDense(nn.Module):

    def __init__(self, num_class = 5, factor = 0.2):
        super(handCNNDense, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 17, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(17),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1))
        self.layer2= nn.Sequential(
            nn.Conv2d(20, 10, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(30, 10, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer4 = nn.Sequential(
            nn.Conv2d(40, 32, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer5 = nn.Linear(32 * 96 * 128, 32)
        self.fc = nn.Linear(32, num_class)
        
    def forward(self, x):
        input_layer1 = x
        output_layer1 = self.layer1(input_layer1)
        input_layer2 = torch.cat((input_layer1, output_layer1), 1)
        output_layer2 = self.layer2(input_layer2)
        input_layer3 = torch.cat((input_layer2, output_layer2), 1)
        output_layer3 = self.layer3(input_layer3)
        input_layer4 = torch.cat((input_layer3, output_layer3), 1)
        out = self.layer4(input_layer4)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.layer5(out))
        out = self.fc(out)

        return out

class handCNN(nn.Module):

    def __init__(self, num_class = 5, factor = 0.1):
        super(handCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1))
        self.layer2= nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.number_pooling_layer = 4
        self.layer5 = nn.Linear(int(480 * 640 * 32 * factor ** 2 // ((2 ** (self.number_pooling_layer - 1)) ** 2)), 32)
        self.fc = nn.Linear(32, num_class)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = F.sigmoid(self.layer5(out))
        out = self.fc(out)

        return out