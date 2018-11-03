import torchvision.transforms as transforms
import torchvision.transforms.functional as F

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

def loadData(train_dir, test_dir, batch_size):
    train_dataset = torchvision.datasets.ImageFolder(root=(train_dir), 
                                                     transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.ImageFolder(root=(test_dir), 
                                                    transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False)

    return train_loader, test_loader
