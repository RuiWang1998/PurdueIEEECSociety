import torch
import torchvision
import torch.nn as nn

# this is a simple model where the number of input channels are 3, the kernel size should be 2-by-2 and the stride is 1, where the padding should only be adjusted according to the kernel size
# input of this model should be times of size 48 * 64 * 3, a colored, down sampling group
# this model contains 4 convolution-BatchNorm-ReLU-MaxPool block and one linear layer
class handCNNDense(nn.Module):

    def __init__(self, num_class = 5, factor = 0.2, k = 10):
        super(handCNNDense, self).__init__()
        self.growth_rate = int(k)

        # the first dense block
        self.layer0= nn.Sequential(
            nn.Conv2d(3, self.growth_rate, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.growth_rate, self.growth_rate, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1))
        self.layer2= nn.Sequential(
            nn.Conv2d(2 * self.growth_rate, self.growth_rate, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(3 * self.growth_rate, self.growth_rate, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))

        self.transition = nn.Sequential(
            nn.Conv2d(self.growth_rate, self.growth_rate, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))

        # the second dense block
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.growth_rate, self.growth_rate, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer5 = nn.Sequential(
            nn.Conv2d(2 * self.growth_rate, self.growth_rate, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer6 = nn.Sequential(
            nn.Conv2d(3 * self.growth_rate, self.growth_rate, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classification = nn.Sequential(
            nn.Conv2d(self.growth_rate, self.growth_rate, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # linear layers
        self.layer7 = nn.Linear(int(self.growth_rate * 96 * 128 / 16), 48)
        self.fc = nn.Linear(48, num_class)
        
    def forward(self, x):

        input_layer0 = x
        output_layer0 = self.layer0(input_layer0)
        output_layer1 = self.layer1(output_layer0)
        input_layer2 = torch.cat((output_layer0, output_layer1), 1)
        output_layer2 = self.layer2(input_layer2)
        input_layer3 = torch.cat((input_layer2, output_layer2), 1)
        output_layer3 = self.layer3(input_layer3)

        out_transition = self.transition(output_layer3)

        output_layer4 = self.layer4(out_transition)
        input_layer5 = torch.cat((out_transition, output_layer4), 1)
        output_layer5 = self.layer5(input_layer5)
        input_layer6 = torch.cat((input_layer5, output_layer5), 1)
        output_layer6 = self.layer6(input_layer6)

        output = self.classification(output_layer6)
        
        output = output_layer6.reshape(output.size(0), -1)
        out = torch.sigmoid(self.layer7(output))
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
        out = torch.sigmoid(self.layer5(out))
        out = self.fc(out)

        return out