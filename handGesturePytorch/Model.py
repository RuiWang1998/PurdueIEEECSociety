import torch
import torchvision
import torch.nn as nn
import torch.functional as F

class handCNNDense(nn.Module):

    def __init__(self, num_class = 5, factor = 0.1, k = 10):
        super(handCNNDense, self).__init__()
        self.growth_rate = int(k)
        # the input layer
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, self.growth_rate, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.denseLayer0 = nn.Sequential(
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.Conv2d(self.growth_rate, self.growth_rate, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.Conv2d(self.growth_rate, self.growth_rate, kernel_size=3, stride=1, padding=1))
        self.denseLayer1 = nn.Sequential(
            nn.BatchNorm2d(2 * self.growth_rate),
            nn.ReLU(),
            nn.Conv2d(2 * self.growth_rate, 2 * self.growth_rate, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * self.growth_rate),
            nn.ReLU(),
            nn.Conv2d(2 * self.growth_rate, self.growth_rate, kernel_size=3, stride=1, padding=1))
        self.denseLayer2 = nn.Sequential(
            nn.BatchNorm2d(3 * self.growth_rate),
            nn.ReLU(),
            nn.Conv2d(3 * self.growth_rate, 3 * self.growth_rate, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3 * self.growth_rate),
            nn.ReLU(),
            nn.Conv2d(3 * self.growth_rate, self.growth_rate, kernel_size=3, stride=1, padding=1))
        # transition layers
        self.transition = nn.Sequential(
            nn.Conv2d(self.growth_rate, self.growth_rate, kernel_size = 2, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1))
        self.linear = nn.Sequential(
            nn.Linear(184320, 100),
            nn.ReLU(),
            nn.Linear(100, num_class))

    def _denseLayer(self, x, layer_idx):
        if layer_idx == 0:
            output = self.denseLayer0(x)
        if layer_idx == 1:
            output = self.denseLayer1(x)
        if layer_idx == 2:
            output = self.denseLayer2(x)

        return output

    def _denseBlock(self, x, layer_num):
        for i in range(layer_num):
            output = self._denseLayer(x, i)
            if i != layer_num - 1:
                x = torch.cat((x, output), 1)

        return output

    def forward(self, x):
        x = self.layer0(x)

        output = self._denseBlock(x, 3)
        x = self.transition(output)
        output = self._denseBlock(x, 3)
        x = self.transition(output)
        output = self._denseBlock(x, 3)

        output = output.reshape(output.size(0), -1)
        output = self.linear(output)

        return output

class handCNN(nn.Module):
    '''
    
    this is a simple model where the number of input channels are 3, the kernel size should be 2-by-2 and the stride is 1, where the padding should only be adjusted according to the kernel size 
    input of this model should be times of size 48 * 64 * 3, a colored, down sampling group this model contains 4 convolution-batchnorm-relu-maxpool block and one linear layer
    '''

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
        # self.layer5 = nn.Linear(int(480 * 640 * 32 * factor ** 2 // ((2 ** (self.number_pooling_layer - 1)) ** 2)), 200)
        self.layer5 = nn.Linear(6144, 200)
        self.fc = nn.Linear(200, num_class)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer5(out)
        out = torch.sigmoid(out)
        out = self.fc(out)

        return out

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self, margin=1.0):

        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        pdist = nn.PairwiseDistance()
        euclidean_distance = pdist(output1, output2)
        loss_contrastive = torch.mean((1-label).float() * torch.pow(euclidean_distance, 2) + label.float() / (euclidean_distance / 100))

        return loss_contrastive