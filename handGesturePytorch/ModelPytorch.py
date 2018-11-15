import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.densenet as Dense

# In this model the densenet architecture will be explored
def handDenseNet(num_init_features = 64, num_classes = 5, growth_rate = 32, block_config=(6,), **kwargs):
    model = Dense.DenseNet(num_init_features = 64, growth_rate = 32, block_config = block_config, num_classes = num_classes, **kwargs)

    return model

# this is a simple model where the number of input channels are 3, the kernel size should be 2-by-2 and the stride is 1, where the padding should only be adjusted according to the kernel size
# input of this model should be times of size 48 * 64 * 3, a colored, down sampling group
# this model contains 4 convolution-BatchNorm-ReLU-MaxPool block and one linear layer
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