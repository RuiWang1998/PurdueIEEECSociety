import torch
import torchvision

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import torchvision.transforms as transforms

from constants import TRAIN_FOLDER, TEST_FOLDER, ALL_FOLDER, TRAIN_AUG, DATA_SOURCE, IMAGE_DIR, BATCH_SIZE, SOURCE

generic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    #transforms.CenterCrop(size=128),
    transforms.Lambda(lambda x: (48, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (1, 1, 1)),
    transforms.ToPILImage(),
])

new_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (1, 1, 1)),
    transforms.ToTensor(),
    transforms.ToPILImage()
    ])


print("loading the data")

train_dataset = torchvision.datasets.ImageFolder(root=(SOURCE + DATA_SOURCE + TRAIN_AUG +'/'), 
                                                     transform=transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(root=(SOURCE + DATA_SOURCE + TEST_FOLDER +'/'), 
                                                    transform=transforms.ToTensor())
all_dataset = torchvision.datasets.ImageFolder(root=(SOURCE + DATA_SOURCE + ALL_FOLDER +'/'), 
                                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False)
all_loader = torch.utils.data.DataLoader(dataset=all_dataset,
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False)

print("data loading finished")